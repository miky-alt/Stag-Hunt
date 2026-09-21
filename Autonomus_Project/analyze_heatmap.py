import os
import glob
import numpy as np

class HeatmapAggregator:
    """
    Classe dedicata all'estrazione, aggregazione e calcolo statistico
    delle Heatmap e degli Eventi Comportamentali dai log CSV dell'ambiente Stag Hunt.
    """
    
    def __init__(self, log_dir="./logs_csv", map_size=(5, 5)):
        self.log_dir = log_dir
        self.map_size = map_size
        
        # Initialize global matrices to zero.
        self.total_a = np.zeros(map_size, dtype=int)
        self.total_b = np.zeros(map_size, dtype=int)
        self.processed_files = 0

        # Initialize global event counters.
        self.total_stags = 0
        self.total_plants = 0
        self.total_maulings = 0

    def aggregate_logs(self):
        """
        Scansiona la cartella dei log e processa tutti i file CSV presenti.
        """
        pattern = os.path.join(self.log_dir, "*.csv")
        lista_file = glob.glob(pattern)

        if not lista_file:
            print(f"[!] No CSV files found in directory '{self.log_dir}'.")
            return

        for file_path in lista_file:
            self._extract_data(file_path)

        print(f"\n[OK] Aggregation completed! Read {self.processed_files} CSV files.")
        self.print_global_statistics()

    def _extract_data(self, file_path):
        """
        Metodo interno per parsare il singolo CSV ed estrarre sia gli eventi che le heatmap.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            idx_a, idx_b = -1, -1
            
            for i, line in enumerate(lines):
                # Extract event counters.
                if "Totale Cervi Catturati:" in line:
                    value = int(line.split(',')[1].strip())
                    self.total_stags += value
                elif "Totale Piante Mangiate:" in line:
                    value = int(line.split(',')[1].strip())
                    self.total_plants += value
                elif "Totale Mauling Subiti:" in line:
                    value = int(line.split(',')[1].strip())
                    self.total_maulings += value
                    
                # Ricerca ancore Heatmap
                elif "--- HEATMAP ESPLORAZIONE AGENTE A ---" in line:
                    idx_a = i
                elif "--- HEATMAP ESPLORAZIONE AGENTE B ---" in line:
                    idx_b = i

            # Extract matrices.
            if idx_a != -1 and idx_b != -1:
                for r in range(self.map_size[0]):
                    values_a = [int(x.strip()) for x in lines[idx_a + 1 + r].split(',')]
                    self.total_a[r] += values_a
                    
                    values_b = [int(x.strip()) for x in lines[idx_b + 1 + r].split(',')]
                    self.total_b[r] += values_b

                self.processed_files += 1
                
        except Exception as e:
            print(f"[ERROR] Unable to read or parse {file_path}: {e}")

    def print_global_statistics(self):
        """
        Formatta e stampa a schermo tutti i totali e le medie.
        """
        # Calculate averages while avoiding division by zero.
        average_stags = self.total_stags / self.processed_files if self.processed_files > 0 else 0
        average_plants = self.total_plants / self.processed_files if self.processed_files > 0 else 0
        average_maulings = self.total_maulings / self.processed_files if self.processed_files > 0 else 0

        print("\n==================================================")
        print("          RIASSUNTO COMPORTAMENTALE GLOBALE       ")
        print("==================================================")
        print(f" 🦌 Totale Cervi Catturati : {self.totale_cervi} (Media/Episodio: {media_cervi:.1f})")
        print(f" 🌱 Totale Piante Mangiate : {self.totale_piante} (Media/Episodio: {media_piante:.1f})")
        print(f" 💥 Totale Mauling Subiti  : {self.totale_mauling} (Media/Episodio: {media_mauling:.1f})")

        # Print heatmaps.
        sum_a = np.sum(self.total_a)
        sum_b = np.sum(self.total_b)

        print("\n==================================================")
        print("    HEATMAP GLOBALE AGENTE A (Percentuali)        ")
        print("==================================================")
        if sum_a > 0:
            percentage_a = (self.total_a / sum_a) * 100
            self._print_grid(percentage_a)
        else:
            print("[!] Nessun dato di movimento registrato per l'Agente A.")

        print("\n==================================================")
        print("    HEATMAP GLOBALE AGENTE B (Percentuali)        ")
        print("==================================================")
        if sum_b > 0:
            percentage_b = (self.total_b / sum_b) * 100
            self._print_grid(percentage_b)
        else:
            print("[!] Nessun dato di movimento registrato per l'Agente B.")
            
        print("==================================================\n")

    def _print_grid(self, percentage_grid):
        for row in percentage_grid:
            formatted_row = [f"{value:5.1f}%" for value in row]
            print(" | ".join(formatted_row))


if __name__ == "__main__":
    analyzer = HeatmapAggregator(log_dir="./logs_csv", map_size=(5, 5))
    analyzer.aggregate_logs()