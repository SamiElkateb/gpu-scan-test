test:
	python -m unittest test-project-gpu.py
	
quicktest:
	python3 test-project-gpu.py --no-large-arrays --no-mem-check 

benchmark:
	python bench.py
	
quickbench:
	python bench.py --fast
