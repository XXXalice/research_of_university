from puckage import Puckage

def main():
    p = Puckage()
    name = p.img_collect()
    p.makemodel(name)

if __name__ == '__main__':
    main()