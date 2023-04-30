test:
	python -m unittest test-project-gpu.py
	
quicktest:
	python3 test-project-gpu.py --no-large-arrays --no-mem-check 

benchmark:
	python benchmark.py
	
quickbench:
	python benchmark.py --fast
