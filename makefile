bindings:
	cd ./scripts/modeldb/bindings && \
	python3 setup.py install && \
	cd ../../../../

ppl_merged:
	python3 -m scripts.modeldb.main_pq \
	-f llama-2-7b.json \
	--dataset wikitext-2-raw-v1 \
	-M 32 \
	--nbits 10 \
	-m \
	--half \
	-p baseline sampling training evaluation

ppl_non_merged:
	python3 -m scripts.modeldb.main_pq \
	-f llama-2-7b.json \
	--dataset wikitext-2-raw-v1 \
	-M 32 \
	--nbits 12 \
	--half \
	-p sampling training evaluation

quant_llama2:
	python3 -m scripts.modeldb.main_pq \
	-f llama-2-7b.json \
	--dataset wikitext-2-raw-v1 \
	-M 64 \
	--nbits 8 \
	--half \
	-p quarot baseline training evaluation \
	--model meta-llama/Llama-2-7b-hf --rotate --a_bits 4 --w_bits 4 --w_clip --w_rtn --save_qmodel_path "./qmodels/llama-2-7b-q-gptq.pth"

llama2_13b:
	python3 -m scripts.modeldb.main_pq \
	-f llama-2-13b.json \
	--dataset wikitext-2-raw-v1 \
	-M 64 \
	--nbits 8 \
	--half \
	-p gptq sampling training evaluation \
	--model "./models/llama-2-13b-hf" --a_bits 4 --a_groupsize 128 --w_bits 4 --w_groupsize 128 --w_clip --save_qmodel_path "./qmodels/llama-2-13b-q-gptq.pth"
	# -p baseline sampling training evaluation

gpt2:
	python3 -m scripts.modeldb.main_pq \
	-f gpt2-xl.json \
	--dataset wikitext-2-raw-v1 \
	-M 32 \
	--nbits 8 \
	--half \
	-p gptq sampling training evaluation \
	--model "./models/gpt2-xl" --a_bits 4 --a_groupsize 128 --w_bits 4 --w_groupsize 128 --w_clip --save_qmodel_path "./qmodels/gpt2-xl-gptq.pth"

mpt:
	python3 -m scripts.modeldb.main_pq \
	-f mpt-7b.json \
	--dataset wikitext-2-raw-v1 \
	-M 64 \
	--nbits 8 \
	--half \
	-p gptq sampling training evaluation \
	--model "./models/mpt-7b" --a_bits 4 --a_groupsize 128 --w_bits 4 --w_groupsize 128 --w_clip --save_qmodel_path "./qmodels/mpt-7b-gptq.pth"

qwen_moe:
	python3 -m scripts.modeldb.main_pq \
	-f qwen1.5-moe-a2.7b.json \
	--dataset wikitext-2-raw-v1 \
	-M 64 \
	--nbits 8 \
	--half \
	-p gptq sampling training evaluation \
	--model "./models/Qwen1.5-MoE-A2.7B" --a_bits 4 --a_groupsize 128 --w_bits 4 --w_groupsize 128 --w_clip --save_qmodel_path "./qmodels/qwen1.5-moe-a2.7b-gptq.pth"


e2e:
	python3 -m scripts.modeldb.main_pq \
	-f longchat-7b.json \
	--dataset _synthetic \
	-M 64 \
	--nbits 8 \
	-m \
	--half \
	-p evaluation

breakdown:
	python3 -m scripts.modeldb.main_pq \
	-f llama-3.1-8b.json \
	--dataset _synthetic \
	-M 64 \
	--nbits 8 \
	-m \
	--half \
	--breakdown \
	-p baseline evaluation

longbench:
	python3 -m scripts.modeldb.main_pq \
	--dataset triviaqa  \
	-f llama-3.1-8b.json \
	-M 64 \
	--nbits 8 \
	-m \
	--half \
	-p baseline sampling training evaluation

debug:
	cuda-gdb --args python3 -m scripts.modeldb.main_pq \
	-f llama-2-7b.json \
	--dataset _synthetic \
	-M 64 \
	--nbits 8 \
	-m \
	-p evaluation



