import cv2
import os
import argparse

def crop_square_no_black_bars(frame, x, y, w, h, padding=0.4):
    """
    Recorta un recuadro cuadrado (1:1) sin deformar ni generar bordes negros.
    """
    img_h, img_w, _ = frame.shape
    
    cx, cy = x + w // 2, y + h // 2
    max_dim = max(w, h)
    side = int(max_dim * (1 + 2 * padding))
    side = min(side, img_h, img_w)
    
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side
    
    if x1 < 0:
        x2 = min(img_w, x2 - x1)
        x1 = 0
    if y1 < 0:
        y2 = min(img_h, y2 - y1)
        y1 = 0
    if x2 > img_w:
        x1 = max(0, x1 - (x2 - img_w))
        x2 = img_w
    if y2 > img_h:
        y1 = max(0, y1 - (y2 - img_h))
        y2 = img_h
        
    return frame[y1:y2, x1:x2]

def load_face_cascade():
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(cascade_path)
    
    if cascade.empty():
        local_xml = "haarcascade_frontalface_default.xml"
        if os.path.exists(local_xml):
            cascade = cv2.CascadeClassifier(local_xml)
            
    if cascade.empty():
        raise RuntimeError("No se pudo cargar el archivo XML de haarcascade.")
        
    return cascade

def extract_frames_from_video(video_path, output_folder, prefix, face_cascade, extract_every_n_frames=60, img_size=(224, 224), padding=0.4):
    if not os.path.exists(video_path):
        print(f"  [OMITIDO] No existe el archivo: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir el video: {video_path}")
        return

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    saved_count = 0
    skipped_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % extract_every_n_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                skipped_count += 1
                frame_count += 1
                continue
            
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            square_crop = crop_square_no_black_bars(frame, x, y, w, h, padding=padding)
            final_img = cv2.resize(square_crop, img_size)
            
            filename = f"{prefix}_{saved_count:05d}.jpg"
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, final_img)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"  ✔ Finalizado. Guardadas {saved_count} imágenes en '{output_folder}' ({skipped_count} ignoradas por falta de rostro).")

def main():
    parser = argparse.ArgumentParser(description="Procesamiento en lote de videos por sujeto.")
    parser.add_argument("-i", "--input_dir", type=str, default=".", help="Carpeta raíz de los videos de origen (default: '.')")
    parser.add_argument("-o", "--output_dir", type=str, default="./dataset", help="Carpeta raíz de destino (default: './dataset')")
    parser.add_argument("--start", type=int, default=1, help="ID del sujeto inicial (default: 1)")
    parser.add_argument("--end", type=int, default=60, help="ID del sujeto final (default: 60)")
    parser.add_argument("-f", "--step", type=int, default=60, help="Frecuencia de extracción de frames (default: 60)")
    parser.add_argument("-s", "--size", type=int, default=224, help="Resolución final cuadrada (default: 224)")
    parser.add_argument("--padding", type=float, default=0.4, help="Margen extra alrededor del rostro (default: 0.4)")

    args = parser.parse_args()

    face_cascade = load_face_cascade()

    # Mapeo: nombre de video de entrada -> carpeta de salida
    video_mapping = {
        "0.mp4": "normal",
        "10.mp4": "drowsy"
    }

    print(f"=== Iniciando proceso en lote (Sujetos {args.start} a {args.end}) ===")

    for subject_id in range(args.start, args.end + 1):
        subject_folder = str(subject_id)
        print(f"\n▶ Sujeto {subject_id}:")

        for video_name, category in video_mapping.items():
            video_path = os.path.join(args.input_dir, subject_folder, video_name)
            output_folder = os.path.join(args.output_dir, subject_folder, category)
            prefix = f"p{subject_id}_{category}"

            print(f"  -> Procesando video: {video_path} ({category})...")
            extract_frames_from_video(
                video_path=video_path,
                output_folder=output_folder,
                prefix=prefix,
                face_cascade=face_cascade,
                extract_every_n_frames=args.step,
                img_size=(args.size, args.size),
                padding=args.padding
            )

    print("\n=== Procesamiento completado con éxito ===")

if __name__ == "__main__":
    main()