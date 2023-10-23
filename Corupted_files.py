def img_validator(source):
    files = get_paths(source)  # A list of complete paths to each image
    invalid_files = []
    for img in files:
        try:
            im = Image.open(img)
            im.verify()
            im.close()
        except (IOError, OSError, Image.DecompressionBombError):
            invalid_files.append(img)

     # Write invalid_files to file