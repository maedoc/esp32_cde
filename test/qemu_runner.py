#!/usr/bin/env python3

import subprocess
import sys
import time
import os
import select
import argparse
import signal
import uuid
import re
import concurrent.futures
import io

import tempfile
import shutil

def run_qemu_interaction(test_input, timeout=60, list_mode=False, output_stream=None):
    if output_stream is None:
        output_stream = sys.stdout.buffer

    def log(msg):
        if isinstance(msg, str):
            output_stream.write((msg + "\n").encode('utf-8', errors='replace'))
        else:
            output_stream.write(msg)

    # Generate unique filenames for flash and efuse for parallel runs
    test_id_str = str(test_input) if test_input else "list"
    unique_id = uuid.uuid4().hex
    
    temp_dir = tempfile.gettempdir()
    flash_bin_path = os.path.join(temp_dir, f"kumo-{test_id_str}-{unique_id}-flash.bin")
    efuse_bin_path = os.path.join(temp_dir, f"kumo-{test_id_str}-{unique_id}-efuse.bin")

    # esptool.py command to merge binaries
    # Determine which binary to use. For 'test' project, it's typically 'build/kmtest.bin' or project name 'build/esp32_cde.bin'
    # Based on previous build output, our binary is 'build/esp32_cde.bin'
    # Wait, the unit test tutorial says 'build/kmtest.bin'.
    # But our project builds 'build/esp32_cde.bin' when we run 'idf.py build'.
    # However, for UNIT TESTING, we usually run 'idf.py -p test build' or similar.
    # The user built the MAIN project with 'idf.py build'.
    # But unit tests usually live in a separate app or the test app.
    # If using 'idf.py test', it builds a separate binary.
    
    # Check if build/kmtest.bin exists, otherwise try build/esp32_cde.bin
    app_bin = "build/kmtest.bin"
    if not os.path.exists(app_bin):
        if os.path.exists("build/esp32_cde.bin"):
            app_bin = "build/esp32_cde.bin"
        elif os.path.exists("build/test_app.bin"):
             app_bin = "build/test_app.bin"
        else:
            # Fallback to user provided logic if I can't find it?
            # Or just assume standard test build
            pass 

    esptool_cmd = [
        "esptool.py", "--chip=esp32", "merge_bin", # Changed to esp32 (not c3) based on project
        f"--output={flash_bin_path}", "--fill-flash-size=4MB",
        "--flash_mode", "dio", "--flash_freq", "40m", "--flash_size", "4MB",
        "0x1000", "build/bootloader/bootloader.bin",
        "0x10000", app_bin,
        "0x8000", "build/partition_table/partition-table.bin",
        # "0x181000", "build/ota_data_initial.bin" # Might not be present in simple test
    ]
    
    # Check if ota data exists
    if os.path.exists("build/ota_data_initial.bin"):
         esptool_cmd.extend(["0x181000", "build/ota_data_initial.bin"])

    log(f"[Runner] Generating flash image: {' '.join(esptool_cmd)}")
    try:
        # Run esptool.py from the current directory (test/)
        result = subprocess.run(esptool_cmd, check=True, cwd=".", capture_output=True)
        log("[Runner] Flash image generated successfully.")
    except subprocess.CalledProcessError as e:
        log(f"[Runner] Error generating flash image: {e}")
        if e.stdout:
            log(f"esptool.py stdout:\n{e.stdout.decode()}")
        if e.stderr:
            log(f"esptool.py stderr:\n{e.stderr.decode()}")
        return 1

    # Prepare efuse binary (optional for QEMU sometimes, but good practice)
    # Using generic blank if not found
    if not os.path.exists(efuse_bin_path):
        with open(efuse_bin_path, "wb") as f:
            f.write(b"\xff" * 128) # ESP32 efuse is smaller/different? Just minimal blank.

    # qemu-system-xtensa command (ESP32)
    # We need to find the qemu binary
    qemu_bin = "qemu-system-xtensa"
    # Check env or path
    
    qemu_cmd = [
        qemu_bin, "-nographic",
        "-machine", "esp32",
        "-drive", f"file={flash_bin_path},if=mtd,format=raw",
        "-serial", "mon:stdio"
    ]
    
    log(f"Starting QEMU: {' '.join(qemu_cmd)}")
    
    # Use a session leader to be able to kill the whole group
    process = subprocess.Popen(
        qemu_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False, # Binary mode for safer non-blocking I/O
        bufsize=0, # Unbuffered
        start_new_session=True,
        cwd="." # Ensure commands are run in the current directory (test/)
    )

    start_time = time.time()
    last_read_time = time.time() # Track last data received time
    output_buffer = b"" # Binary buffer
    
    prompt_seen = False
    input_sent = False
    test_finished = False
    exit_code = 1 # Default to failure
    
    # Non-blocking read setup
    os.set_blocking(process.stdout.fileno(), False)
    
    try:
        while True:
            if time.time() - start_time > timeout:
                log(f"\nTIMEOUT ({timeout}s) reached!")
                break

            # Check if process is still running
            if process.poll() is not None:
                log("\nProcess exited prematurely.")
                break

            # Read available data in binary mode
            ready = select.select([process.stdout], [], [], 0.1)
            if ready[0]:
                chunk = os.read(process.stdout.fileno(), 1024) # Use os.read for binary
                if not chunk: # EOF
                    log("\n[Runner] QEMU closed stdout (EOF).")
                    break
                
                last_read_time = time.time() # Update last read time
                output_stream.write(chunk) # Write binary chunk directly
                output_stream.flush()
                
                output_buffer += chunk
                # Decode safely for logic checks
                try:
                    full_text = output_buffer.decode('utf-8', errors='ignore')
                except:
                    full_text = ""

                if not prompt_seen:
                    # Check for prompt - Unity test menu
                    if "Press ENTER" in full_text or "Tests" in full_text and "Enter" in full_text:
                        prompt_seen = True
                        time.sleep(0.5)
                        
                        if list_mode:
                            log("\n[Runner] Sending ENTER to list tests...")
                            process.stdin.write(b"\n")
                        else:
                            # Unity menu expects "[id]" or "name"
                            # If test_input is just the ID number
                            log(f"\n[Runner] Sending test ID: {test_input}...")
                            process.stdin.write(f"\"{test_input}\"\n".encode()) 
                        process.stdin.flush()
                        input_sent = True

                if input_sent:
                    if list_mode:
                        # Success if we see the prompt again OR if we have silence for a while after listing
                        if full_text.count("Press ENTER") > 1 or "Test name" in full_text:
                             exit_code = 0
                             break
                    else:
                        if "Test ran in" in full_text or "Tests 0 Failures 0 Ignored" in full_text:
                            test_finished = True
                        
                        if test_finished:
                            if "0 Failures" in full_text:
                                log("\n[Runner] SUCCESS detected.")
                                exit_code = 0
                                break
                            elif "Failures" in full_text and "0 Failures" not in full_text:
                                if re.search(r'\b[1-9][0-9]* Failures', full_text):
                                     log("\n[Runner] FAILURE detected.")
                                     exit_code = 1
                                     break
            else:
                # No data ready
                if list_mode and input_sent:
                     # If we are in list mode, sent input, and haven't seen data for 0.5s, assume list is done
                     if time.time() - last_read_time > 1.0:
                         log("\n[Runner] Silence detected in list mode. Assuming success.")
                         exit_code = 0
                         break

    except KeyboardInterrupt:
        log("\nInterrupted by user.")
    finally:
        log("\n[Runner] Terminating QEMU and cleaning up temporary files...")
        
        # Try killing the process group first
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass # Process already gone

        # Also try killing the process directly just in case
        if process.poll() is None:
            process.terminate()

        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            log("[Runner] Force killing QEMU...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                 process.kill()

        # Clean up temporary files
        if os.path.exists(flash_bin_path):
            try:
                os.remove(flash_bin_path)
            except OSError as e:
                log(f"[Runner] Warning: Could not remove flash bin: {e}")

        if os.path.exists(efuse_bin_path):
             try:
                os.remove(efuse_bin_path)
             except OSError as e:
                log(f"[Runner] Warning: Could not remove efuse bin: {e}")
    
    return exit_code

def get_all_tests(timeout=30):
    capture = io.BytesIO()
    exit_code = run_qemu_interaction(None, timeout=timeout, list_mode=True, output_stream=capture)
    if exit_code != 0:
        print("Failed to list tests")
        return []
    
    output = capture.getvalue().decode('utf-8', errors='replace')
    tests = []
    # Match pattern: (1)     "test name" [tags]
    pattern = re.compile(r'^\((\d+)\)\s+"(.*?)"', re.MULTILINE)
    for match in pattern.finditer(output):
        tests.append((match.group(1), match.group(2)))
    return tests