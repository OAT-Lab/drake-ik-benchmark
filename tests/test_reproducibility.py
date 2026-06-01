"""
Ensures reproducibility

 Asserts:
    - The same seed produces identical scenarios
    - Scenarios are immutable once generated
    - The capable+available solver set is stable across scenarios. 
"""