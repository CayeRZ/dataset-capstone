## how to run the script
python3 extract.py -i [insert the path] -o ./dataset --start 35 --end 36 -f 90


here are the arguments
parser = argparse.ArgumentParser(description="Procesamiento en lote de videos por sujeto.")
    parser.add_argument("-i", "--input_dir", type=str, default=".", help="Carpeta raíz de los videos de origen (default: '.')")
    parser.add_argument("-o", "--output_dir", type=str, default="./dataset", help="Carpeta raíz de destino (default: './dataset')")
    parser.add_argument("--start", type=int, default=1, help="ID del sujeto inicial (default: 1)")
    parser.add_argument("--end", type=int, default=60, help="ID del sujeto final (default: 60)")
    parser.add_argument("-f", "--step", type=int, default=60, help="Frecuencia de extracción de frames (default: 60)")
    parser.add_argument("-s", "--size", type=int, default=224, help="Resolución final cuadrada (default: 224)")
    parser.add_argument("--padding", type=float, default=0.4, help="Margen extra alrededor del rostro (default: 0.4)")