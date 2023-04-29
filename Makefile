test:
	python -m unittest test-project-gpu.py

benchmark:
	python bench.py

fastbench:
	python bench.py --fast
