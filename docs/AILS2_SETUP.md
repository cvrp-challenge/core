# AILS2 Solver Setup Guide

## Overview

AILS2 is a Java-based solver that requires:
1. Java JDK installed
2. The AILS2 JAR file compiled/built
3. The solver wrapper (already done ✅)

## Step 1: Install Java

### On Ubuntu/WSL:
```bash
sudo apt update
sudo apt install openjdk-17-jdk

# Verify installation
java -version
javac -version
```

### On macOS:
```bash
brew install openjdk@17
```

### On Windows:
Download and install from: https://adoptium.net/

## Step 2: Update AILS2 Submodule

Make sure the AILS2 submodule is fully checked out:

```bash
cd solver/ails2
git submodule update --init --recursive
```

Or from the repo root:
```bash
git submodule update --init --recursive solver/ails2
```

## Step 3: Build the JAR File

### Option A: Use the Build Script (Recommended)

```bash
cd solver/ails2
./build.sh
```

This will:
- Check for Java installation
- Find all Java source files
- Compile them
- Create `AILSII.jar`

### Option B: Manual Build

If the build script doesn't work, try manual compilation:

```bash
cd solver/ails2

# Create build directory
mkdir -p build/classes

# Find and compile all Java files
find src -name "*.java" > sources.txt
javac -d build/classes -encoding UTF-8 @sources.txt

# Create JAR (main class is SearchMethod.AILSII)
cd build/classes
jar cfe ../../AILSII.jar SearchMethod.AILSII .
cd ../..
```

### Option C: Download Pre-built JAR

Check if there's a pre-built JAR available:
- Check the GitHub releases: https://github.com/vinymax10/AILS-CVRP/releases
- Check the INFORMS repository: https://github.com/INFORMSJoC/2023.0106

## Step 4: Verify the Build

Test that the JAR works:

```bash
cd solver/ails2
java -jar AILSII.jar -file data/X-n110-k13.vrp -rounded true -best 0 -limit 10 -stoppingCriterion Time
```

If this runs without errors, you're good to go!

## Step 5: Test Integration

Test the Python integration:

```python
from master.routing.solver import solve

result = solve(
    instance="X-n110-k13.vrp",
    solver="ails2",
    solver_options={
        "max_runtime": 10.0,
        "rounded": True
    }
)
print(f"Cost: {result.cost}, Runtime: {result.runtime}s")
```

## Troubleshooting

### "Java not found"
- Install Java JDK (not just JRE)
- Make sure `java` and `javac` are in your PATH
- Restart your terminal after installation

### "No Java source files found"
- The submodule might not be fully checked out
- Run: `git submodule update --init --recursive solver/ails2`
- Check if `src/` directory exists: `ls -la solver/ails2/src/`

### "Could not find or load main class"
- The main class might be different
- Check the source code for the actual main class name
- Update the build script with the correct main class

### "JAR file not found" when running solver
- Make sure `AILSII.jar` exists in `solver/ails2/`
- Or specify the path manually: `solver_options={"ails2_jar": "path/to/AILSII.jar"}`

## File Locations

The solver wrapper will look for the JAR in these locations (in order):
1. `solver/ails2/AILSII.jar` (root)
2. `solver/ails2/build/AILSII.jar`
3. `solver/ails2/target/AILSII.jar`
4. `solver/ails2/dist/AILSII.jar`
5. `solver/ails2/bin/AILSII.jar`

Or you can override with: `solver_options={"ails2_jar": "/path/to/jar"}`

## Next Steps

Once AILS2 is built and working:
- You can use it in your routing code: `solver="ails2"`
- It supports cluster subproblems (like FILO)
- It supports both time limits (`max_runtime`) and iteration limits (`no_improvement`)
