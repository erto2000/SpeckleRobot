import paramiko


def transfer_file(local_path, remote_path, hostname, username, password, port=22):
    try:
        # Create an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the Raspberry Pi
        ssh_client.connect(hostname, port, username, password)

        # Create an SFTP session
        sftp = ssh_client.open_sftp()

        unix_path = remote_path.replace("\\", "/")

        # Transfer the file
        sftp.put(local_path, unix_path)
        print(f"File '{local_path}' successfully transferred to '{unix_path}'")

        # Close the SFTP session and SSH connection
        sftp.close()
        ssh_client.close()

    except Exception as e:
        print(f"Error transferring file: {e}")
