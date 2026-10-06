import cv2
import os
import argparse

def extract_frames(video_path, output_folder, extract_every_n_frames=30, prefix="frame"):
    """
    Extrae fotogramas de un video y los guarda con un prefijo personalizado.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el archivo de video: {video_path}")
        return
        
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % extract_every_n_frames == 0:
            # Construcción del nombre del archivo con el prefijo
            filename = f"{prefix}_{saved_count:05d}.jpg"
            output_path = os.path.join(output_folder, filename)
            
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"Proceso finalizado. Se guardaron {saved_count} imágenes en '{output_folder}' con el prefijo '{prefix}'.")

if __name__ == "__main__":
    # Configuración de los argumentos para la consola
    parser = argparse.ArgumentParser(description="Extrae fotogramas de un video para entrenamiento de Computer Vision.")
    
    parser.add_argument("-v", "--video", required=True, help="Ruta al archivo de video de entrada")
    parser.add_argument("-o", "--output", required=True, help="Directorio donde se guardarán las imágenes")
    parser.add_argument("-f", "--step", type=int, default=30, help="Frecuencia de extracción (cada N frames). Default: 30")
    parser.add_argument("-p", "--prefix", type=str, default="frame", help="Prefijo para las imágenes guardadas. Default: 'frame'")
    
    args = parser.parse_args()
    
    extract_frames(
        video_path=args.video,
        output_folder=args.output,
        extract_every_n_frames=args.step,
        prefix=args.prefix
    )