import cv2
import os
import glob
import subprocess
import time
import sys

# === CONFIGURARE CĂI ȘI SCRIPTURI ===
INPUT_FOLDER = "images"
MODEL_NAME   = "mono+stereo_640x192"

# Numele scripturilor tale
SCRIPT_MONODEPTH = "test_simple.py"
SCRIPT_CONVERT   = "convert_disp_to_txt.py"
SCRIPT_PLOT      = "demonstratie_calitate.py"
APP_CUDA         = "./scale_app"

def run_command(cmd):
    """
    Rulează o comandă în terminal. Returnează True dacă a mers, False dacă a crăpat.
    """
    try:
        # > /dev/null ascunde output-ul din terminal ca să nu fie spam
        subprocess.check_call(cmd + " > /dev/null 2>&1", shell=True)
    except subprocess.CalledProcessError:
        print(f"[EROARE] Comanda a eșuat: {cmd}")
        return False
    return True

def show_image_timed(image_path, duration_sec, window_name, fullscreen=False):
    """
    Afișează imaginea pentru 'duration_sec' secunde.
    Forteaza dimensiunea ferestrei daca imaginea e mica.
    """
    if not os.path.exists(image_path):
        print(f"[SKIP] Nu găsesc imaginea pentru afișare: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        return

    # 1. Creăm fereastra cu opțiunea de redimensionare
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    if fullscreen:
        # Modul Fullscreen exclusiv
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        # === FIX: FORȚĂM O DIMENSIUNE MARE PENTRU CELELALTE FERESTRE ===
        # Setăm o dimensiune generoasă (ex: 1200 lățime x 600 înălțime)
        # Astfel, chiar dacă depth map-ul e mic (640x192), se va vedea mare.
        cv2.resizeWindow(window_name, 1200, 600)
        
        # Opțional: Mutăm fereastra puțin spre centrul ecranului (x=100, y=100)
        cv2.moveWindow(window_name, 100, 100)
    
    cv2.imshow(window_name, img)
    
    # waitKey primește milisecunde
    key = cv2.waitKey(int(duration_sec * 1000))
    
    cv2.destroyWindow(window_name)
    
    # Dacă apeși ESC (27) sau Q (113) oprește tot scriptul
    if key == 27 or key == 113: 
        raise KeyboardInterrupt

def cleanup_iteration(folder_path, base_name):
    """
    Sterge fisierele generate DOAR pentru imaginea curenta, ca sa nu stearga originalul.
    """
    # Definim pattern-urile fisierelor generate de scripturi
    # Ex: imagine_disp.npy, imagine_disp.jpeg, imagine_dashboard.png, etc.
    patterns = [
        f"{base_name}_disp.npy",
        f"{base_name}_disp.jpeg",      # Monodepth genereaza de obicei si un jpeg vizual
        f"{base_name}_depth_rel.txt",
        f"{base_name}_metric_cuda.txt",
        f"{base_name}_dashboard.png",  # Plotul final
        "metric_depth_cuda.txt",       # Fisier temporar posibil
        "*top_dowb.png",               # Typo-ul specificat
        "*top_down.png"
    ]

    for pat in patterns:
        # Daca e wildcard generic (*), il cautam cu glob, altfel calea directa
        if "*" in pat:
            full_search = os.path.join(folder_path, pat)
            files = glob.glob(full_search)
            for f in files:
                try: os.remove(f)
                except: pass
        else:
            full_path = os.path.join(folder_path, pat)
            if os.path.exists(full_path):
                try: os.remove(full_path)
                except: pass

def get_kitti_images():
    """
    Returneaza lista imaginilor originale, excluzand output-urile vechi ramase.
    """
    # Cautam png si jpg
    all_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.png')) + 
                       glob.glob(os.path.join(INPUT_FOLDER, '*.jpg')))
    
    valid_images = []
    for f in all_files:
        filename = os.path.basename(f)
        # Excludem fisierele care contin cuvinte cheie de output
        if "_disp" in filename: continue
        if "_dashboard" in filename: continue
        if "depth" in filename: continue
        valid_images.append(f)
    
    return valid_images

def main():
    print("=== START LIVE DEMO (Processing + Display + Cleanup) ===")
    
    # 1. Identificare imagini originale
    images = get_kitti_images()
    
    if not images:
        print(f"[STOP] Nu am gasit imagini curate in {INPUT_FOLDER}.")
        sys.exit(1)

    print(f"Am gasit {len(images)} imagini de procesat.")

    # 2. Iterare prin fiecare imagine
    for i, img_path in enumerate(images):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        folder = os.path.dirname(img_path)
        
        print(f"\n[{i+1}/{len(images)}] Procesez: {base_name} ...")

        # --- DEFINIRE NUME FISIERE OUTPUT ---
        # Monodepth genereaza de obicei {nume}_disp.npy si {nume}_disp.jpeg
        f_npy        = os.path.join(folder, f"{base_name}_disp.npy")
        f_disp_vis   = os.path.join(folder, f"{base_name}_disp.jpeg") # Poza Depth Map (vizuala)
        
        f_txt_rel    = os.path.join(folder, f"{base_name}_depth_rel.txt")
        f_txt_metric = os.path.join(folder, f"{base_name}_metric_cuda.txt")
        f_dashboard  = os.path.join(folder, f"{base_name}_dashboard.png") # Poza Plot

        # --- A. EXECUTIE SCRIPTURI (PROCESAREA) ---
        
        # 1. Monodepth (Genereaza disparity map)
        if not run_command(f"python3 {SCRIPT_MONODEPTH} --image_path {img_path} --model_name {MODEL_NAME}"):
            print("Eroare la Monodepth. Trec la urmatoarea.")
            continue

        # 2. Convert NPY -> TXT
        run_command(f"python3 {SCRIPT_CONVERT} --input_npy {f_npy} --output_txt {f_txt_rel}")

        # 3. Scale App (CUDA)
        # Stergem eventualul fisier vechi creat de aplicatia C++
        if os.path.exists("metric_depth_cuda.txt"): 
            os.remove("metric_depth_cuda.txt")
            
        run_command(f"{APP_CUDA} {f_txt_rel} 640 192")
        
        # Redenumim outputul aplicatiei C++
        if os.path.exists("metric_depth_cuda.txt"):
            os.rename("metric_depth_cuda.txt", f_txt_metric)
        else:
            print("[WARN] Scale App nu a generat output-ul. Folosesc date relative.")

        # 4. Generare Plot/Dashboard
        run_command(f"python3 {SCRIPT_PLOT} --image {img_path} --metric {f_txt_metric} --output_img {f_dashboard}")


        # --- B. AFISARE SECVENTIALA ---
        
        # 1. POZA FULLSCREEN (3 secunde) -> Originala
        print(" -> Afisez Original (3s)")
        show_image_timed(img_path, 3, "Original", fullscreen=True)

        # 2. POZA DEPTHMAP (2 secunde) -> Outputul vizual de la Monodepth
        # De obicei test_simple salveaza un _disp.jpeg. Daca nu exista, incercam dashboard-ul
        print(" -> Afisez Depth Map (2s)")
        if os.path.exists(f_disp_vis):
            show_image_timed(f_disp_vis, 2, "Depth Map", fullscreen=False)
        else:
            # Fallback daca nu exista jpeg-ul de depth separat
            show_image_timed(f_npy, 2, "Depth NPY (Eroare afisare)", fullscreen=False)

        # 3. POZA PLOT (4 secunde) -> Dashboard-ul creat de tine
        print(" -> Afisez Plot (6s)")
        show_image_timed(f_dashboard, 6, "Dashboard Plot", fullscreen=False)


        # --- C. CLEANUP (Curatenie) ---
        print(" -> Curatenie fisiere generate...")
        cleanup_iteration(folder, base_name)

    print("\n=== FINAL DEMO ===")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nOprit fortat de utilizator.")
