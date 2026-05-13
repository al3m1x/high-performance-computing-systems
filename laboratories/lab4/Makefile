CC=gcc
CFLAGS=-O2
LIBS=-lm
OMPFLAGS=-fopenmp

ifeq (run,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif

main: main.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o main $(LIBS)

run:
	./main $(RUN_ARGS)

clean:
	rm -f main