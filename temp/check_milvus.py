from pymilvus import connections, utility
from pymilvus.exceptions import MilvusException
import time

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

def check_milvus_status(max_retries=10, retry_interval_sec=5):
    """
    Connects to Milvus and performs a basic check to verify if it's running.
    """
    print(f"Attempting to connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")

    for i in range(max_retries):
        try:
            # 1. Establish connection
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            print(f"Connection attempt {i+1}/{max_retries}: Successfully connected to Milvus.")

            # 2. Perform a basic utility operation (e.g., get server version)
            # This confirms that the Milvus server is not just listening, but also responsive.
            version = utility.get_server_version()
            print(f"Milvus server is running. Version: {version}")
            return True

        except MilvusException as e:
            print(f"Connection attempt {i+1}/{max_retries}: MilvusException: {e}")
            if i < max_retries - 1:
                print(f"Retrying in {retry_interval_sec} seconds...")
                time.sleep(retry_interval_sec)
            else:
                print("Max retries reached. Could not connect to Milvus.")
                return False
        except Exception as e:
            # Catch other potential errors
            print(f"Connection attempt {i+1}/{max_retries}: An unexpected error occurred: {e}")
            if i < max_retries - 1:
                print(f"Retrying in {retry_interval_sec} seconds...")
                time.sleep(retry_interval_sec)
            else:
                print("Max retries reached. Could not connect to Milvus.")
                return False
        finally:
            # Always disconnect to clean up resources, even if connection failed
            try:
                connections.disconnect()
            except Exception:
                pass # Ignore errors on disconnect if not connected

    return False

if __name__ == "__main__":
    if check_milvus_status():
        print("Milvus is confirmed to be running and responsive.")
    else:
        print("Milvus is not running or not responsive.")

