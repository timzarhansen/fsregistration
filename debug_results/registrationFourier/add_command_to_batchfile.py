def add_command_to_batchfile(fullpath, command, append=False):
    """Add a command to a batch file.
    
    Args:
        fullpath: Path to the batch file
        command: Command string to add
        append: If True, append to file; if False, overwrite
    """
    mode = 'a' if append else 'w'
    with open(fullpath, mode) as fid:
        fid.write(f"{command}\n")
