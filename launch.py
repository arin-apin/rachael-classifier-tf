#!/usr/bin/env python3
"""
Simple launcher script for Rachael Classifier Training App
Creates directories, starts docker-compose, and opens browser automatically
"""
import os
import subprocess
import sys
import time
import webbrowser
import requests
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['models', 'data', 'logs']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Directory '{dir_name}' ready")

def check_docker():
    """Check if docker and docker-compose are available"""
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
        print("✅ Docker is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not available. Please install Docker first.")
        sys.exit(1)
    
    # Check for docker-compose or docker compose
    compose_cmd = None
    try:
        subprocess.run(['docker-compose', '--version'], check=True, capture_output=True)
        compose_cmd = ['docker-compose']
        print("✅ docker-compose is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(['docker', 'compose', 'version'], check=True, capture_output=True)
            compose_cmd = ['docker', 'compose']
            print("✅ docker compose is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ docker-compose is not available. Please install Docker Compose.")
            sys.exit(1)
    
    return compose_cmd

def wait_for_service(url="http://localhost:7860", timeout=120):
    """Wait for the service to be ready"""
    print(f"🔄 Waiting for service to start at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ Service is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    
    print(f"\n⚠️  Service didn't start within {timeout} seconds")
    return False

def main():
    print("🚀 Rachael Classifier Training App Launcher")
    print("=" * 50)

    # Create directories
    print("\n📁 Creating directories...")
    create_directories()

    # Check Docker
    print("\n🐳 Checking Docker...")
    compose_cmd = check_docker()

    # Set environment variables for user/group IDs to avoid permission issues
    env = os.environ.copy()
    env['USER_ID'] = str(os.getuid())
    env['GROUP_ID'] = str(os.getgid())

    # Start docker-compose
    print("\n🚀 Starting application...")
    try:
        subprocess.run(compose_cmd + ['up', '-d'], check=True, env=env)
        print("✅ Application containers started!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)
    
    # Wait for service and open browser
    print("\n🌐 Preparing to open browser...")
    if wait_for_service():
        print("🎉 Opening browser...")
        webbrowser.open('http://localhost:7860')
        print("✅ Browser opened successfully!")
        print("\n🎯 Application is ready at: http://localhost:7860")
        print("\n📋 Available commands:")
        print("  • To stop:     python3 launch.py stop")
        print("  • To restart:  python3 launch.py restart")
        print("  • To view logs: docker compose logs -f")
    else:
        print("⚠️  Service might be starting slowly. Please check manually at: http://localhost:7860")

def stop_application():
    """Stop the application"""
    print("🛑 Stopping Rachael Classifier Training App...")
    compose_cmd = check_docker()

    # Set environment variables for user/group IDs
    env = os.environ.copy()
    env['USER_ID'] = str(os.getuid())
    env['GROUP_ID'] = str(os.getgid())

    try:
        subprocess.run(compose_cmd + ['down'], check=True, env=env)
        print("✅ Application stopped successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to stop application: {e}")

def restart_application():
    """Restart the application"""
    print("🔄 Restarting Rachael Classifier Training App...")
    compose_cmd = check_docker()

    # Set environment variables for user/group IDs
    env = os.environ.copy()
    env['USER_ID'] = str(os.getuid())
    env['GROUP_ID'] = str(os.getgid())

    try:
        subprocess.run(compose_cmd + ['restart'], check=True, env=env)
        print("✅ Application restarted successfully!")
        if wait_for_service():
            webbrowser.open('http://localhost:7860')
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to restart application: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "stop":
            stop_application()
        elif command == "restart":
            restart_application()
        elif command == "help":
            print("Usage: python3 launch.py [command]")
            print("Commands:")
            print("  start    - Start the application (default)")
            print("  stop     - Stop the application")
            print("  restart  - Restart the application")
            print("  help     - Show this help message")
        else:
            print(f"Unknown command: {command}")
            print("Use 'python3 launch.py help' for available commands")
    else:
        main()