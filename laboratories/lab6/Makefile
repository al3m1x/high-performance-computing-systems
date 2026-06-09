NVCC=nvcc
CUDAFLAGS=#-arch=sm_35

ifeq (run,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif


main: cuda.cu
	${NVCC} ${CUDAFLAGS} $< -o cuda

run:
	./cuda $(RUN_ARGS)

clean:
	rm cuda
