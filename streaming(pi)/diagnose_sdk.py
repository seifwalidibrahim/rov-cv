#!/usr/bin/env python3
"""
Run on Windows to find the Daheng Galaxy SDK and gxipy.
GXVision installs the SDK alongside itself.
"""
import os
import sys
import subprocess

print("=== Searching for Daheng/GXVision install on Windows ===")

search_paths = [
    r"C:\Program Files\Daheng Imaging",
    r"C:\Program Files (x86)\Daheng Imaging",
    r"C:\Program Files\Galaxy",
    r"C:\Program Files (x86)\Galaxy",
    r"C:\Daheng Imaging",
    r"C:\Galaxy",
]

found_root = None
for path in search_paths:
    if os.path.exists(path):
        print(f"  FOUND: {path}")
        found_root = path
        for root, dirs, files in os.walk(path):
            for f in files:
                if 'gxi' in f.lower() or 'galaxy' in f.lower():
                    print(f"    → {os.path.join(root, f)}")
    else:
        print(f"  not found: {path}")

print()
print("=== Searching for gxipy in Python paths ===")
for p in sys.path:
    candidate = os.path.join(p, 'gxipy')
    if os.path.exists(candidate):
        print(f"  FOUND gxipy at: {candidate}")

print()
print("=== Searching entire C: drive for gxipy folder (may take a moment) ===")
result = subprocess.run(
    ['where', '/R', r'C:\Program Files', 'gxipy*'],
    capture_output=True, text=True
)
if result.stdout:
    print(result.stdout)
else:
    # Try broader search
    result2 = subprocess.run(
        ['dir', '/s', '/b', r'C:\gxipy'],
        capture_output=True, text=True, shell=True
    )
    print(result2.stdout or "gxipy folder not found in Program Files")

print()
print("=== Python version and site-packages ===")
print(f"Python: {sys.version}")
print(f"Site packages:")
import site
for p in site.getsitepackages():
    print(f"  {p}")

print()
print("=== Checking if GxCamera DLL is accessible ===")
dll_paths = [
    r"C:\Program Files\Daheng Imaging\GalaxySDK\APIDll\Win64\GxIAPI.dll",
    r"C:\Windows\System32\GxIAPI.dll",
    r"C:\Windows\SysWOW64\GxIAPI.dll",
]
for dll in dll_paths:
    print(f"  {'FOUND' if os.path.exists(dll) else 'missing'}: {dll}")