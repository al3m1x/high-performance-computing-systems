# High-Performance Computing Systems

## Setup (Windows)

1. Open Ubuntu WSL terminal
2. Navigate to the lab directory ex.:
   ```bash
   cd /mnt/c/Users/juliu/high-performance-computing-systems/lab2
   ```

3. Install required dependencies:
   ```bash
   sudo apt update
   sudo apt install build-essential openmpi-bin libopenmpi-dev
   ```

## Building

```bash
make
```

## Cleaning

```bash
make clean
```

## Running

```bash
make run 1 100000000 L2
```

Example with different parameters:
```bash
make run <param1> <param2> <param3>
``` 