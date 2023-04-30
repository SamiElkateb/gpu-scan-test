test:
	python -m unittest test-project-gpu.py
	
quicktest:
	python3 test-project-gpu.py --no-large-arrays --no-mem-check  --failfast

benchmark:
	python benchmark.py
	
quickbench:
	python benchmark.py --fast

coursetest:
	python test-project-gpu.py --course-only

earlytest:
	python test-project-gpu.py --course-only --no-independent --no-inclusive
