import torch
from pykeops.torch import LazyTensor
from ..utils.Singleton import Singleton

from ..utils.Namespace import UniConfig
from ..utils.Timer import Timer

def l2Ns(l):
    """
    Heuristic to select proper number of splits Ns for a sequence of length l.
    """
    if l > 2048:
        return 32 # Ns32 is manually selected. It's the fastest kernel for long sequences with >2k tokens
        # we are not benefiting from bigger Ns possibly due to hardware limitations
    elif l > 256:
        return 16
    elif l > 128:
        return 4
    elif l > 64:
        return 2
    else:
        return 1 # To small to benefit from parallelization
    
def scalarTypeToStr(scalar_t):
    if scalar_t == torch.float32:
        return 'f32'
    elif scalar_t == torch.float16:
        return 'f16'
    else:
        raise ValueError(f"Unknown scalar type: {scalar_t}")

class KernelRegistry:
    def __init__(self, *, M=64, d=128, nbits=8, nh=32, scalar_t=torch.float32):
        self.kernels = {}
        self.partial_out_buffers = {}
        self.partial_lse_buffers = {}
        self.M = M
        self.d = d
        self.nbits = nbits
        self.nh = nh
        self.scalar_t = scalar_t

    def get_kernel(self, l=4096):
        Ns = l2Ns(l)
        if Ns not in self.kernels:
            self.kernels[Ns] = self.get_custom_kernel_with_allocated_buffer(l)
        return self.kernels[Ns]
    
    def get_custom_kernel_with_allocated_buffer(self, l=4096):
        M = self.M
        d = self.d
        nbits = self.nbits
        nh = self.nh
        scalar_t = self.scalar_t

        bs = 1 # typical for inference

        if nbits != 8:
            raise NotImplementedError("Only uint8 code type is supported for now")
        
        Ns = l2Ns(l)
        scalar_str = scalarTypeToStr(scalar_t)
        try:
            fname = f"flash_decoding_allocated_buffer_{scalar_str}u8_Ns{Ns}Lt{d}d{d}M{M}C{2**nbits}"
            func = getattr(__import__('bindings'), fname)
        except Exception as e:
            print(f"{fname} not compiled. Modify scripts/modeldb/bindings/setup.py to compile it.")
            raise e

        # prepare allocated buffer, don't worry this is small
        # NOTE: This overhead is not introduced by PQ! It's introduced by parallelizing softmax across blocks
        # i.e. the overhead is implementation-specific
        partial_out_buffer = torch.empty(bs, nh, Ns+1, d, dtype=scalar_t, device='cuda')
        partial_lse_buffer = torch.empty(bs, nh, Ns+1, dtype=scalar_t, device='cuda')

        self.partial_out_buffers[Ns] = partial_out_buffer
        self.partial_lse_buffers[Ns] = partial_lse_buffer

        # define a closure so we do not need to specify the buffer every time
        def flash_decoding(query, key_codes, value_codes, key_cents, value_cents, key_residuals, value_residuals, r):
            # partial_lse_buffer.fill_(float('-inf')) # no need to re-initialize
            # partial_out_buffer.fill_(0)
            return func(
                query, 
                key_codes, 
                value_codes, 
                key_cents, 
                value_cents, 
                key_residuals,
                value_residuals,
                r,
                partial_out_buffer, 
                partial_lse_buffer,
            )
        
        return flash_decoding

class DynamicPQCache(metaclass=Singleton):
    def __init__(self, *, bs, nh, num_key_value_heads, M, layer_num, dtype=torch.uint8, nbits=8, d=128, scalar_t=torch.float32, merged_training=False):
        self.bs = bs
        self.nh = nh
        self.num_key_value_heads = num_key_value_heads
        self.M = M
        self.layer_num = layer_num
        self.dtype = dtype
        self.nbits = nbits
        self.d = d
        self.scalar_t = scalar_t
        self.merged_training = merged_training

        self.max_residual_length = d # Common practice: Lt = d to make residual cache square so kernel is easy to implement
        self.registery = KernelRegistry(M=M, d=d, nbits=nbits, nh=nh, scalar_t=scalar_t)

        self.init_cache()

    def init_cache(self):
        self.key_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, 0, self.M), dtype=self.dtype, device='cuda')
            for _ in range(self.layer_num)
        ]

        self.value_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, 0, self.M), dtype=self.dtype, device='cuda')
            for _ in range(self.layer_num)
        ]

        self.key_residual_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, self.max_residual_length, self.d), dtype=self.scalar_t, device='cuda')
            for _ in range(self.layer_num)
        ]

        self.value_residual_cache = [
            torch.zeros((self.bs, self.num_key_value_heads, self.max_residual_length, self.d), dtype=self.scalar_t, device='cuda')
            for _ in range(self.layer_num)
        ]

        self.seen_tokens = [0 for _ in range(self.layer_num)]
        self.residualed_tokens = [0 for _ in range(self.layer_num)]

    def cat_codes(self, key_codes, value_codes, layer_idx):
        """
        Dynamic cache's concatenate is very inefficient. We optimize cat by triggering it only when residual cache is full.
        """
        # naive cat
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_codes], dim=2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_codes], dim=2)
        self.seen_tokens[layer_idx] += key_codes.size(2)

    def set_cent(self, key_cent, value_cent):
        """
        cent is of shape (M, c, d//M) if merged_training=True
        or (layer_num, M, c, d//M) if merged_training=False
        """
        if self.merged_training is False:
            layer_num, M, c, d_m = key_cent.shape
            assert layer_num == self.layer_num, f"layer_num {layer_num} does not match self.layer_num {self.layer_num}"
        else:
            M, c, d_m = key_cent.shape
        assert key_cent.shape == value_cent.shape, "key_cent and value_cent must have the same shape"
        assert d_m * M == self.d, f"d_m * M {d_m * M} must equal to d {self.d}"
        assert c == 2 ** self.nbits, f"c {c} must equal to 2 ** nbits {2 ** self.nbits}"
        
        if key_cent is value_cent:
            self._cent = key_cent.contiguous()
            self.key_cent = self._cent
            self.value_cent = self._cent
        else:
            self.key_cent = key_cent.contiguous()
            self.value_cent = value_cent.contiguous()

            # # expand cent to match batch size and num_heads
            # # Can be avoided if encode use a cumsom kernel that supports broadcasting?
            # self.key_cent_expand = self.key_cent.unsqueeze(0).expand(self.nh, -1, -1, -1).unsqueeze(0).expand(self.bs, -1, -1, -1, -1).contiguous()
            # self.value_cent_expand = self.value_cent.unsqueeze(0).expand(self.nh, -1, -1, -1).unsqueeze(0).expand(self.bs, -1, -1, -1, -1).contiguous()

    def update(self, key_states, value_states, layer_idx, distort_recent=False):
        """
        When not dispatching to custom kernel, this function is used to update the cache.

        Update the cache and return the updated key-value states. This behaviour is inherently compatible with DynamicCache from transformers.
        """
        # states are of shape (bs, num_key_value_heads, update_length, d)
        
        # assert cent is set
        assert hasattr(self, 'key_cent') and hasattr(self, 'value_cent')
    
        current_device = key_states.device
        past_length = self.key_cache[layer_idx].size(2)

        # move cache and cent to the same device as the input
        # We are assuming that the input is on the same device. Can kv be on different devices? IDK
        self.key_cache[layer_idx] = self.key_cache[layer_idx].to(current_device)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].to(current_device)
        self.key_cent = self.key_cent.to(current_device)
        self.value_cent = self.value_cent.to(current_device)
        
        key_cent = self.key_cent if self.merged_training else self.key_cent[layer_idx]
        value_cent = self.value_cent if self.merged_training else self.value_cent[layer_idx]

        if distort_recent:
            key_codes = sa_encode_4d_keops(key_states, key_cent, target_dtype=self.dtype)
            value_codes = sa_encode_4d_keops(value_states, value_cent, target_dtype=self.dtype)
            # self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_codes], dim=2)
            # self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_codes], dim=2)
            self.cat_codes(key_codes, value_codes, layer_idx)

            key_states = sa_decode_4d(self.key_cache[layer_idx], key_cent)
            value_states = sa_decode_4d(self.value_cache[layer_idx], value_cent)
        else:
            if past_length > 0:
                past_key_states = sa_decode_4d(self.key_cache[layer_idx], key_cent)
                past_value_states = sa_decode_4d(self.value_cache[layer_idx], value_cent)

            key_codes = sa_encode_4d_keops(key_states, key_cent, target_dtype=self.dtype)
            value_codes = sa_encode_4d_keops(value_states, value_cent, target_dtype=self.dtype)
            # self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_codes], dim=2)
            # self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_codes], dim=2)
            self.cat_codes(key_codes, value_codes, layer_idx)

            if past_length > 0:
                key_states = torch.cat([past_key_states, key_states], dim=2)
                value_states = torch.cat([past_value_states, value_states], dim=2)

        return key_states, value_states
    
    def prefill(self, query_states, key_states, value_states, layer_idx, distort_recent=False):
        """
        Pre-fill the cache and return attention output.

        For quantization methods, usually prefill phase is done with original data type. Quantized data type is only
        used in the decoding phase. But to show quantization effect on model performance on probability-based benchmarks(for 
        example, perplexity) which only have decoding phase, we leave a switch to use quantized data type in prefill phase.

        TL;DR: Leave distort_recent=False when deploying models.
        """
        key_cent = self.key_cent if self.merged_training else self.key_cent[layer_idx]
        value_cent = self.value_cent if self.merged_training else self.value_cent[layer_idx]
        
        with Timer("DynamicPQCache.prefill"):
            with Timer("DynamicPQCache.prefill.encode"):
                key_codes = sa_encode_4d_keops(key_states, key_cent, target_dtype=self.dtype)
                value_codes = sa_encode_4d_keops(value_states, value_cent, target_dtype=self.dtype)
                torch.cuda.synchronize()

            with Timer("DynamicPQCache.prefill.cat"):
                self.cat_codes(key_codes, value_codes, layer_idx)
                torch.cuda.synchronize()

            if distort_recent is True:
                with Timer("DynamicPQCache.prefill.decode"):
                    key_states = sa_decode_4d(key_codes, key_cent)
                    value_states = sa_decode_4d(value_codes, value_cent)
                torch.cuda.synchronize()

            with Timer("DynamicPQCache.prefill.attention"):
                from torch.nn.functional import scaled_dot_product_attention as sdpa
                from transformers.models.llama.modeling_llama import repeat_kv

                num_heads = query_states.size(1)
                num_kv_heads = key_states.size(1)
                nrep = num_heads // num_kv_heads

                key_states = repeat_kv(key_states, nrep)
                value_states = repeat_kv(value_states, nrep)
                
                return sdpa(query_states, key_states, value_states, is_causal=True)

    def residual_attention(self, query_states, layer_idx):
        """
        Deprecated: implemented by kernel

        Attention with residual cache. 
        """

        key_states = self.key_residual_cache[layer_idx]
        value_states = self.value_residual_cache[layer_idx]
        scale = self.d ** -0.5

        S = scale * query_states @ key_states.transpose(-2, -1) # (bs, num_heads, 1, max_residual_length)
        # mask non-residual tokens in S to -inf(in self.scalar_t)
        S[:, :, :, self.residualed_tokens[layer_idx]:] = torch.tensor(float('-inf'), dtype=self.scalar_t, device='cuda')
        out = torch.softmax(S, dim=-1) @ value_states # (bs, num_heads, 1, d)
        lse = torch.logsumexp(S, dim=-1, keepdim=False) # (bs, num_heads, 1)

        return out, lse

    def decoding(self, query_states, key_states, value_states, layer_idx):
        """
        
        """
        breakdown = UniConfig().breakdown
        key_cent = self.key_cent if self.merged_training else self.key_cent[layer_idx]
        value_cent = self.value_cent if self.merged_training else self.value_cent[layer_idx]

        # flush residual to cache if needed
        r = self.residualed_tokens[layer_idx]
        if r == self.max_residual_length:
            with Timer("LlamaSdpaAttention.forward.flush"):
                with Timer("LlamaSdpaAttention.forward.flush.encode"):
                    key_codes = sa_encode_4d_keops(self.key_residual_cache[layer_idx], key_cent, target_dtype=self.dtype)
                    value_codes = sa_encode_4d_keops(self.value_residual_cache[layer_idx], value_cent, target_dtype=self.dtype)
                    if breakdown is True: torch.cuda.synchronize()

                with Timer("LlamaSdpaAttention.forward.flush.cat"):
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_codes], dim=2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_codes], dim=2)
                    if breakdown is True: torch.cuda.synchronize()

                self.residualed_tokens[layer_idx] = 0
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.copy"):
            # add states to residual
            r = self.residualed_tokens[layer_idx]
            n = key_states.size(2)
            self.key_residual_cache[layer_idx][:, :, r: r + n, :] = key_states
            self.value_residual_cache[layer_idx][:, :, r: r + n, :] = value_states
            self.residualed_tokens[layer_idx] += n
            self.seen_tokens[layer_idx] += n
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.kernel"):
            kernel = self.registery.get_kernel(l=self.seen_tokens[layer_idx])
            out = kernel(
                query_states,
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                key_cent,
                value_cent,
                self.key_residual_cache[layer_idx],
                self.value_residual_cache[layer_idx],
                self.residualed_tokens[layer_idx]
            )
            if breakdown is True: torch.cuda.synchronize()
        
        return out
    
        ######################################## DEBUGGING ########################################

        # manually combine self.registery.partial_out_buffers[32] and self.registery.partial_lse_buffers[32]
        partial_outs = self.registery.partial_out_buffers[32] # (bs, nh, Lt+1, d)
        partial_lses = self.registery.partial_lse_buffers[32] # (bs, nh, Lt+1)
        # Perform log-sum-exp on partial_lses to get normalization factor
        L = torch.max(partial_lses, dim=2, keepdim=True)[0]  # (bs, nh, 1) max along Lt+1 dimension
        scaling_denom = torch.sum(torch.exp(partial_lses - L), dim=2, keepdim=True)  # (bs, nh, 1)

        # Compute the weighted sum of partial_outs using the scaling factors
        merged_output = torch.sum(
            partial_outs * torch.exp(partial_lses - L)[:, :, :, None] / scaling_denom[:, :, :, None], 
            dim=2, keepdim=True
        )

        # manually combine non-residual part
        partial_outs = partial_outs[:, :, :-1, :] # (bs, nh, Lt, d)
        partial_lses = partial_lses[:, :, :-1] # (bs, nh, Lt)
        # Perform log-sum-exp on partial_lses to get normalization factor
        L = torch.max(partial_lses, dim=2, keepdim=True)[0]  # (bs, nh, 1) max along Lt dimension
        scaling_denom = torch.sum(torch.exp(partial_lses - L), dim=2, keepdim=True)  # (bs, nh, 1)

        pq_out = torch.sum(
            partial_outs * torch.exp(partial_lses - L)[:, :, :, None] / scaling_denom[:, :, :, None],
            dim=2, keepdim=True
        )
        residual_out = self.registery.partial_out_buffers[32][:, :, -1:, :] # (bs, nh, 1, d)
        residual_lse = self.registery.partial_lse_buffers[32][:, :, -1:] # (bs, nh, 1)

        # manually verify the correctness of the output
        K_hat = sa_decode_4d(self.key_cache[layer_idx], self.key_cent)
        V_hat = sa_decode_4d(self.value_cache[layer_idx], self.value_cent)
        K = torch.cat([K_hat, self.key_residual_cache[layer_idx][:, :, :self.residualed_tokens[layer_idx], :]], dim=2)
        V = torch.cat([V_hat, self.value_residual_cache[layer_idx][:, :, :self.residualed_tokens[layer_idx], :]], dim=2)
        
        pq_out_gt = torch.nn.functional.scaled_dot_product_attention(query_states, K_hat, V_hat)
        residual_out_gt, residual_lse_gt = self.residual_attention(query_states, layer_idx)

        manual_out = torch.nn.functional.scaled_dot_product_attention(query_states, K, V)

        pq_mean_diff = (pq_out - pq_out_gt).abs().mean().item()
        residual_mean_diff = (residual_out - residual_out_gt).abs().mean().item()
        mean_diff = (out - manual_out).abs().mean().item()
        
        if pq_mean_diff > 1e-3:
            print(f"Layer {layer_idx} PQ Mean diff:", (pq_out - pq_out_gt).abs().mean().item())
        if residual_mean_diff > 1e-3:
            print(f"Layer {layer_idx} Residual Mean diff:", (residual_out - residual_out_gt).abs().mean().item())
        if mean_diff > 1e-3:
            print(f"Layer {layer_idx} Mean diff:", (out - manual_out).abs().mean().item())

        return out

    @property
    def pq_cache_size(self):
        return sum([cache.numel() * cache.element_size() for cache in self.key_cache + self.value_cache])

    @property
    def codebook_size(self):
        if hasattr(self, '_cent'):
            return self._cent.numel() * self._cent.element_size()
        else:
            k_cent_size = self.key_cent.numel() * self.key_cent.element_size()
            v_cent_size = self.value_cent.numel() * self.value_cent.element_size()
            return k_cent_size + v_cent_size

    @property
    def residual_cache_size(self):
        return sum([cache.numel() * cache.element_size() for cache in self.key_residual_cache + self.value_residual_cache])
    
    @property
    def registry_size(self):
        partial_out_size = sum([
            buffer.numel() * buffer.element_size() for buffer in self.registery.partial_out_buffers.values()
        ])
        partial_lse_size = sum([
            buffer.numel() * buffer.element_size() for buffer in self.registery.partial_lse_buffers.values()
        ])
        return partial_out_size + partial_lse_size

def sa_encode_4d(X, C, target_dtype=torch.uint8):
    """
    Find the nearest codeword for each vector in X among the rows of C.
    Returns the indices of the nearest rows.

    X is shaped (bs, num_heads, n, d) then reshaped to (bs, num_heads, n, M, d//M)
    C is shaped (M, c, d//M) # same codebook for all heads
        or already expanded to (bs, num_heads, M, c, d_m)

    returns (bs, num_heads, n, M)

    Note: num_heads refers to num_key_value_heads in GQA
    """

    bs, num_heads, n, d = X.shape

    if len(C.shape) == 3:
        M, c, d_m = C.shape
        C = C.unsqueeze(0).unsqueeze(0).expand(bs, num_heads, -1, -1, -1)  # (bs, num_heads, M, c, d_m)
    else:
        bs, num_heads, M, c, d_m = C.shape

    # Reshape X and C for batch computation
    X = X.reshape(bs, num_heads, n, M, d_m)

    # X: (bs * num_heads * M, n, d_m)
    X = X.permute(0, 1, 3, 2, 4).reshape(bs * num_heads * M, n, d_m)
    # C: (bs * num_heads * M, c, d_m)
    C = C.reshape(bs * num_heads * M, -1, d_m)

    # Compute distances
    dis = torch.cdist(X, C, p=2)  # (bs * num_heads * M, n, c)

    # Reshape distances back
    dis = dis.reshape(bs, num_heads, M, n, c).permute(0, 1, 3, 2, 4)  # (bs, num_heads, n, M, c)

    # Get codes
    codes = torch.argmin(dis, dim=-1)  # (bs, num_heads, n, M)

    return codes.contiguous().to(target_dtype)

def sa_encode_4d_keops(X, C, target_dtype=torch.uint8):
    """
    Find the nearest codeword for each vector in X among the rows of C using PyKeOps.
    Returns the indices of the nearest codewords.

    X is shaped (bs, num_heads, n, d) then reshaped to (bs, num_heads, n, M, d//M)
    C is shaped (M, c, d//M) # same codebook for all heads
        or already expanded to (bs, num_heads, M, c, d_m)

    Returns (bs, num_heads, n, M)
    """

    bs, num_heads, n, d = X.shape

    if len(C.shape) == 3:
        M, c, d_m = C.shape
        C = C.unsqueeze(0).unsqueeze(0).expand(bs, num_heads, -1, -1, -1)  # (bs, num_heads, M, c, d_m)
    else:
        bs, num_heads, M, c, d_m = C.shape
    
    d_m = d // M
    # Reshape X to (bs, num_heads, n, M, d_m)
    X = X.reshape(bs, num_heads, n, M, d_m)

    # Reshape X and C for batch computation
    # X: (bs * num_heads * M, n, d_m)
    X = X.permute(0, 1, 3, 2, 4).reshape(bs * num_heads * M, n, d_m)
    # C: (bs * num_heads * M, c, d_m)
    C = C.reshape(bs * num_heads * M, -1, d_m)

    # ensure contiguous and convert to float32
    # https://github.com/getkeops/keops/issues/393
    X = X.contiguous().to(torch.float32)
    C = C.contiguous().to(torch.float32)

    # Use PyKeOps LazyTensors for efficient computation
    x_i = LazyTensor(X[:, :, None, :])  # (B, n, 1, d_m)
    c_j = LazyTensor(C[:, None, :, :])  # (B, 1, c, d_m)

    # Compute squared Euclidean distances
    D_ij = ((x_i - c_j) ** 2).sum(-1)  # (B, n, c)

    # Get indices of the nearest codewords
    indices = D_ij.argmin(dim=2)  # (B, n)

    # Reshape indices back to (bs, num_heads, n, M)
    indices = indices.view(bs, num_heads, M, n).permute(0, 1, 3, 2)

    return indices.contiguous().to(target_dtype)

def sa_decode_4d(codes, C):
    """
    Reconstruct the vectors from the codebook C using the indices in codes.

    C is shaped (num_heads, M, c, d//M) # different codebooks for each head
            or  (M, c, d//M) # same codebook for all heads

    codes is shaped (bs, num_heads, n, M)

    returns (bs, num_heads, n, d)
    """
    bs, num_heads, n, M = codes.shape

    if len(C.shape) == 3:
        M, c, d_m = C.shape
        C = C.unsqueeze(0).unsqueeze(0).expand(bs, num_heads, -1, -1, -1)  # (bs, num_heads, M, c, d_m)
    else:
        bs, num_heads, M, c, d_m = C.shape

    if len(C.shape) == 3:
        C = C.unsqueeze(0)  # Now C is (1, M, c, d_m)

    codes = codes.to(torch.long)
    # Expand codes to (bs, num_heads, n, M, 1)
    codes_expanded = codes.unsqueeze(-1)  # (bs, num_heads, n, M, 1)

    # Expand codes to match d_m dimension
    codes_expanded = codes_expanded.expand(-1, -1, -1, -1, d_m)  # (bs, num_heads, n, M, d_m)

    # Expand C to match n dimension
    C_expanded = C.unsqueeze(2).expand(-1, -1, n, -1, -1, -1)  # (bs, num_heads, n, M, c, d_m)

    # Gather codewords
    decoded = torch.gather(C_expanded, 4, codes_expanded.unsqueeze(-2))  # (bs, num_heads, n, M, 1, d_m)
    decoded = decoded.squeeze(4)  # (bs, num_heads, n, M, d_m)

    # Reshape decoded to (bs, num_heads, n, M * d_m)
    decoded = decoded.reshape(bs, num_heads, n, M * d_m)

    return decoded

def nbits2dtype(nbits):
    if nbits <= 8:
        return torch.uint8
    elif nbits <= 16:
        return torch.uint16
    elif nbits <= 32:
        return torch.uint32
    elif nbits <= 64:
        return torch.uint64
    else:
        raise ValueError("nbits must be <= 64")

def train_opq(X, M, nbits, niter=25) -> torch.Tensor:
    """
    X: np.ndarray (n, d)

    A: torch.Tensor (d, d)
    cent: torch.Tensor (M, 2**nbits, d//M)

    opq_index.sa_encode(Y) is equivalent to index.sa_encode(Y @ A.T)
    """
    import numpy as np
    import faiss

    n, d = X.shape

    assert d > M and d % M == 0, "d must be divisible by M"

    opq_matrix = faiss.OPQMatrix(d, M)
    index = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)
    index.pq.cp.niter = niter
    index.pq.cp.verbose = True
    index.verbose = False
    opq_index = faiss.IndexPreTransform(opq_matrix, index)
    opq_index.train(X)

    A = faiss.vector_to_array(opq_matrix.A).reshape(d, d)
    A = torch.from_numpy(A)

    cent = faiss.vector_to_array(index.pq.centroids).reshape(M, 2**nbits, d//M)
    cent = torch.from_numpy(cent)

    return A, cent

def train_pq(X, M, nbits, niter=25) -> torch.Tensor:
    """
    X: np.ndarray (n, d)

    cent: torch.Tensor (M, 2**nbits, d//M)
    """

    import numpy as np
    import faiss

    n, d = X.shape

    assert d > M and d % M == 0, "d must be divisible by M"

    index = faiss.IndexPQ(d, M, nbits)
    index.pq.cp.niter = niter
    index.pq.cp.verbose = True
    index.verbose = False
    index.train(X)

    cent = faiss.vector_to_array(index.pq.centroids).reshape(M, 2**nbits, d//M)
    cent = torch.from_numpy(cent)

    return cent
