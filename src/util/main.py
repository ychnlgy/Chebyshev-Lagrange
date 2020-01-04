MAIN = "__main__"

def main(name=None):

    def main_wrapper(prog):
    
        if name == MAIN:
            
            import sys
            args = dict([d.split("=") for d in sys.argv[1:]])
            prog(**args)
        
        return prog
    
    if callable(name):
        prog = name
        name = MAIN
        return main_wrapper(prog)
    else:
        return main_wrapper
