import pip
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])  

if __name__ == '__main__':
    import_or_install('nibabel')
    import_or_install('requests')
    import_or_install('multiprocessing')
    import_or_install('tensorflow')
    import_or_install('keras')
    import_or_install('opencv-python')
    import_or_install('matplotlib')
    import_or_install('numpy')
    import_or_install('scipy')
    import_or_install('scikit-image')



