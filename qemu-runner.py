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
    esptool_cmd = [
        "esptool.py", "--chip=esp32c3", "merge_bin",
        f"--output={flash_bin_path}", "--fill-flash-size=4MB",
        "--flash_mode", "dio", "--flash_freq", "80m", "--flash_size", "4MB",
        "0x0", "build/bootloader/bootloader.bin",
        "0x190000", "build/kmtest.bin",
        "0x9000", "build/partition_table/partition-table.bin",
        "0x181000", "build/ota_data_initial.bin"
    ]

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

    # Prepare efuse binary
    original_efuse_path = "build/qemu_efuse.bin"
    if os.path.exists(original_efuse_path):
        log(f"[Runner] Using existing efuse image: {original_efuse_path}")
        shutil.copyfile(original_efuse_path, efuse_bin_path)
    else:
        log("[Runner] Creating blank efuse image")
        with open(efuse_bin_path, "wb") as f:
            f.write(b"\xff" * 0x200) # ESP32-C3 efuse is 512 bytes

    # qemu-system-riscv32 command
    qemu_cmd = [
        "qemu-system-riscv32", "-M", "esp32c3",
        "-drive", f"file={flash_bin_path},if=mtd,format=raw",
        "-drive", f"file={efuse_bin_path},if=none,format=raw,id=efuse",
        "-global", "driver=nvram.esp32c3.efuse,property=drive,value=efuse",
        "-global", "driver=timer.esp32c3.timg,property=wdt_disable,value=true",
        "-nic", "user,model=open_eth", "-nographic",
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
                full_text = output_buffer.decode('utf-8', errors='replace')

                if not prompt_seen:
                    # Check for prompt
                    if "Press ENTER" in full_text:
                        prompt_seen = True
                        time.sleep(0.5)
                        
                        if list_mode:
                            log("\n[Runner] Sending ENTER to list tests...")
                            process.stdin.write(b"\n")
                        else:
                            log(f"\n[Runner] Sending test ID: {test_input}...")
                            process.stdin.write(f"{test_input}\n".encode())
                        process.stdin.flush()
                        input_sent = True

                if input_sent:
                    if list_mode:
                        # Success if we see the prompt again OR if we have silence for a while after listing
                        if full_text.count("Press ENTER") > 1:
                             exit_code = 0
                             break
                    else:
                        if "Test ran in" in full_text:
                            test_finished = True
                        
                        if test_finished:
                            if "Tests 0 Failures" in full_text:
                                log("\n[Runner] SUCCESS detected.")
                                exit_code = 0
                                break
                            elif "Failures" in full_text and "Tests 0 Failures" not in full_text:
                                # Double check it's not "0 Failures"
                                # We can look for regex " [1-9][0-9]* Failures"
                                if re.search(r'\b[1-9][0-9]* Failures', full_text):
                                     log("\n[Runner] FAILURE detected.")
                                     exit_code = 1
                                     break
            else:
                # No data ready
                if list_mode and input_sent:
                     # If we are in list mode, sent input, and haven't seen data for 0.5s, assume list is done
                     if time.time() - last_read_time > 0.5:
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

def run_single_test(test_id, timeout):
    capture = io.BytesIO()
    exit_code = run_qemu_interaction(test_id, timeout=timeout, output_stream=capture)
    return test_id, exit_code, capture.getvalue()

def main():
    parser = argparse.ArgumentParser(description='QEMU Test Runner')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_list = subparsers.add_parser('list', help='List available tests')
    parser_list.add_argument('--timeout', type=int, default=30, help='Timeout in seconds')

    parser_run = subparsers.add_parser('run', help='Run tests')
    parser_run.add_argument('test_ids', nargs='*', help='Test IDs to run')
    parser_run.add_argument('--all', action='store_true', help='Run all available tests')
    parser_run.add_argument('--jobs', '-j', type=int, default=1, help='Number of concurrent jobs')
    parser_run.add_argument('--timeout', type=int, default=60, help='Timeout per test in seconds')

    args = parser.parse_args()

    if args.command == 'list':
        sys.exit(run_qemu_interaction(None, timeout=args.timeout, list_mode=True))
    elif args.command == 'run':
        tests_to_run = []
        if args.all:
            print("Fetching test list...")
            all_tests = get_all_tests(timeout=args.timeout)
            print(f"Found {len(all_tests)} tests.")
            tests_to_run = [t[0] for t in all_tests]
        elif args.test_ids:
            tests_to_run = args.test_ids
        else:
            print("Error: Specify test IDs or --all")
            sys.exit(1)

        print(f"Running {len(tests_to_run)} tests with {args.jobs} jobs...")
        
        failures = []
        passed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_test = {
                executor.submit(run_single_test, tid, args.timeout): tid 
                for tid in tests_to_run
            }
            
            for future in concurrent.futures.as_completed(future_to_test):
                tid = future_to_test[future]
                try:
                    _, code, output_bytes = future.result()
                    if code == 0:
                        print(f"PASS: Test {tid}")
                        passed += 1
                    else:
                        print(f"FAIL: Test {tid}")
                        failures.append(tid)
                        # Print the output of the failed test
                        print(f"--- Output for Test {tid} ---")
                        print(output_bytes.decode('utf-8', errors='replace'))
                        print("---------------------------")
                except Exception as exc:
                    print(f"Test {tid} generated an exception: {exc}")
                    failures.append(tid)

        print("\nSummary:")
        print(f"Passed: {passed}")
        print(f"Failed: {len(failures)}")
        if failures:
            print(f"Failed Tests: {', '.join(failures)}")
            sys.exit(1)
        sys.exit(0)

if __name__ == '__main__':
    main()
