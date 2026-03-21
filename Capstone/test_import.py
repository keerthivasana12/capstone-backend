import traceback
try:
    import api.main
    print("Import successful!")
except Exception as e:
    with open("err.txt", "w") as f:
        f.write(traceback.format_exc())
    print("Import failed, traceback written to err.txt")
