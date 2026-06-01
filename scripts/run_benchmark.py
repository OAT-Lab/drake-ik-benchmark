"""
The thin CLI conductor. 
- Contains no benchmark logic: 
    - load config
    - build plant
    - generate scenarios
    - loop over each (scenario + solver) pair through the single runner path
    - collect results
    - save
Orchestration only — all real logic lives in the src package.
"""