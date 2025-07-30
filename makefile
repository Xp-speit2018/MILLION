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



