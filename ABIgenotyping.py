import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from fpdf import FPDF
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

# ====================== CONFIGURAÇÃO DE ESTILO ======================
class StyleConfig:
    PRIMARY_COLOR = "#2E86AB"
    SECONDARY_COLOR = "#A23B72"
    SUCCESS_COLOR = "#18A558"
    WARNING_COLOR = "#F39C12"
    DANGER_COLOR = "#E74C3C"
    LIGHT_BG = "#343A40"
    DARK_BG = "#343A40"
    FONT_FAMILY = "Segoe UI"
    
    @classmethod
    def configure_styles(cls):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Primary.TButton', 
                       background=cls.PRIMARY_COLOR, 
                       foreground='white',
                       font=(cls.FONT_FAMILY, 10, 'bold'))
        
        style.configure('Success.TButton',
                       background=cls.SUCCESS_COLOR,
                       foreground='white',
                       font=(cls.FONT_FAMILY, 10, 'bold'))
        
        style.configure('Custom.TTreeview',
                       background=cls.LIGHT_BG,
                       fieldbackground=cls.LIGHT_BG,
                       font=(cls.FONT_FAMILY, 9))
        
        style.configure('Title.TLabel',
                       font=(cls.FONT_FAMILY, 14, 'bold'),
                       foreground=cls.PRIMARY_COLOR)

# ====================== FUNÇÕES SSR (CODOMINANTE) ======================

def validate_ssr_data(df, id_col=0):
    """Valida dados SSR e identifica amostras com apenas um alelo"""
    issues = []
    allele_cols = df.columns.drop(df.columns[id_col])
    
    if len(allele_cols) % 2 != 0:
        issues.append("Número de colunas de alelos não é par.")
        return issues
    
    for i in range(0, len(allele_cols), 2):
        marker_name = allele_cols[i].split()[0]
        allele1 = df[allele_cols[i]]
        allele2 = df[allele_cols[i + 1]]
        
        # Verificar amostras com apenas um alelo
        single_alleles = ((pd.notna(allele1) & pd.isna(allele2)) | 
                         (pd.isna(allele1) & pd.notna(allele2)))
        
        if single_alleles.any():
            samples_with_single = df.iloc[single_alleles.values, id_col].tolist()
            issues.append(f"Marcador {marker_name}: {len(samples_with_single)} amostra(s) com apenas um alelo serão tratadas como homozigotas")
    
    return issues

def process_codominant_data(df, id_col=0):
    """Processa dados SSR codominantes - CORRIGIDO para tratar alelos únicos como homozigotos"""
    df_result = pd.DataFrame()
    df_result['SampleID'] = df.iloc[:, id_col]

    allele_cols = df.columns.drop(df.columns[id_col])
    if len(allele_cols) % 2 != 0:
        raise ValueError("Número de colunas de alelos deve ser par.")

    for i in range(0, len(allele_cols), 2):
        allele1 = df[allele_cols[i]]
        allele2 = df[allele_cols[i + 1]]
        marker_name = allele_cols[i].split()[0]

        # CORREÇÃO: Tratar alelos únicos como homozigotos
        processed_allele1 = []
        processed_allele2 = []
        
        for a1, a2 in zip(allele1, allele2):
            a1_rounded = round(a1) if pd.notnull(a1) else np.nan
            a2_rounded = round(a2) if pd.notnull(a2) else np.nan
            
            # Se apenas um alelo está presente, tratar como homozigoto
            if pd.notnull(a1_rounded) and pd.isnull(a2_rounded):
                processed_allele1.append(a1_rounded)
                processed_allele2.append(a1_rounded)  # Replicar o mesmo alelo
            elif pd.isnull(a1_rounded) and pd.notnull(a2_rounded):
                processed_allele1.append(a2_rounded)  # Replicar o mesmo alelo
                processed_allele2.append(a2_rounded)
            else:
                processed_allele1.append(a1_rounded)
                processed_allele2.append(a2_rounded)
        
        df_result[f"{marker_name}_1"] = processed_allele1
        df_result[f"{marker_name}_2"] = processed_allele2

    return df_result

def calculate_ssr_indices(df, id_col=0):
    """Calcula índices genéticos para marcadores SSR - CORRIGIDO para dados homozigotos"""
    result_summary = []
    per_marker_data = []

    allele_cols = [col for col in df.columns if col != 'SampleID']
    
    # Agrupar colunas por marcador
    markers = {}
    for col in allele_cols:
        if col.endswith('_1') or col.endswith('_2'):
            marker_name = col.rsplit('_', 1)[0]  # Remove _1 ou _2 do final
            if marker_name not in markers:
                markers[marker_name] = []
            markers[marker_name].append(col)
    
    total_alleles_set = set()
    total_ho_sum = 0
    total_he_sum = 0
    total_f_sum = 0
    total_pic_sum = 0
    valid_markers = 0

    for marker_name, allele_cols in markers.items():
        if len(allele_cols) != 2:
            continue
            
        allele1_col, allele2_col = allele_cols
        allele1 = df[allele1_col]
        allele2 = df[allele2_col]
        
        # Coletar todos os alelos válidos
        all_alleles = []
        heterozygotes = 0
        samples_count = 0
        
        for a1, a2 in zip(allele1, allele2):
            if pd.notna(a1) and pd.notna(a2):
                all_alleles.extend([a1, a2])
                if a1 != a2:
                    heterozygotes += 1
                samples_count += 1
        
        if samples_count == 0:
            continue
            
        # Calcular frequências alélicas
        allele_counts = {}
        for allele in all_alleles:
            allele_counts[allele] = allele_counts.get(allele, 0) + 1
        
        total_alleles = len(all_alleles)
        freqs = {allele: count / total_alleles for allele, count in allele_counts.items()}
        
        # Cálculo de He (heterozigosidade esperada)
        he = 1 - sum(f ** 2 for f in freqs.values())
        
        # Cálculo de Ho (heterozigosidade observada)
        ho = heterozygotes / samples_count
        
        # Cálculo de F (coeficiente de endogamia)
        f = (he - ho) / he if he != 0 else 0
        
        # Cálculo de Ae (número efetivo de alelos)
        ae = 1 / (1 - he) if he < 1 else np.nan
        
        # Cálculo de PIC (Conteúdo de Polimorfismo Informacional)
        sum_f2 = sum(f**2 for f in freqs.values())
        sum_f4 = sum(f**4 for f in freqs.values())
        pic = 1 - sum_f2 - sum_f2**2 + sum_f4

        total_ho_sum += ho
        total_he_sum += he
        total_f_sum += f
        total_pic_sum += pic
        valid_markers += 1
        total_alleles_set.update(allele_counts.keys())

        per_marker_data.append({
            "Locus": marker_name,
            "N": samples_count,
            "A": len(allele_counts),
            "Ae": round(ae, 3),
            "Ho": round(ho, 3),
            "He": round(he, 3),
            "F": round(f, 3),
            "PIC": round(pic, 3)
        })

    if valid_markers > 0:
        overall_he = total_he_sum / valid_markers
        summary = {
            "Locus": "OVERALL",
            "N": "-",
            "A": len(total_alleles_set),
            "Ae": round(1 / (1 - overall_he), 3) if overall_he < 1 else np.nan,
            "Ho": round(total_ho_sum / valid_markers, 3),
            "He": round(overall_he, 3),
            "F": round(total_f_sum / valid_markers, 3),
            "PIC": round(total_pic_sum / valid_markers, 3)
        }
        result_summary.append(summary)

    per_marker_df = pd.DataFrame(per_marker_data)
    summary_df = pd.DataFrame(result_summary)
    return per_marker_df, summary_df

def save_ssr_indices_txt(df, filename):
    """Salva resultados SSR em arquivo TXT"""
    df.to_csv(filename, sep='\t', index=False)

def save_ssr_indices_pdf(df, filename):
    """Salva resultados SSR em arquivo PDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="SSR Genetic Indices", ln=True, align='C')
    pdf.ln(10)

    col_widths = [30, 15, 15, 15, 15, 15, 15, 15]
    headers = ["Locus", "N", "A", "Ae", "Ho", "He", "F", "PIC"]
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1)
    pdf.ln()

    for _, row in df.iterrows():
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, str(row[header]), 1)
        pdf.ln()

    pdf.output(filename)

def create_ssr_visualizations(df, output_prefix):
    """Cria visualizações para dados SSR"""
    try:
        # Gráfico de resumo dos índices
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('SSR Genetic Indices Summary', fontsize=16, fontweight='bold')
        
        # Dados para plotting (excluindo a linha "OVERALL")
        plot_df = df[df['Locus'] != 'OVERALL']
        
        if len(plot_df) == 0:
            return False
            
        # Gráfico 1: Número de alelos por locus
        axes[0,0].bar(plot_df['Locus'], plot_df['A'], color=StyleConfig.PRIMARY_COLOR, alpha=0.7)
        axes[0,0].set_title('Number of Alleles per Locus')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Ho e He por locus
        x_pos = np.arange(len(plot_df['Locus']))
        width = 0.35
        axes[0,1].bar(x_pos - width/2, plot_df['Ho'], width, label='Ho', color=StyleConfig.SUCCESS_COLOR, alpha=0.7)
        axes[0,1].bar(x_pos + width/2, plot_df['He'], width, label='He', color=StyleConfig.SECONDARY_COLOR, alpha=0.7)
        axes[0,1].set_title('Observed vs Expected Heterozygosity')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(plot_df['Locus'], rotation=45)
        axes[0,1].legend()
        
        # Gráfico 3: Coeficiente de endogamia
        axes[1,0].bar(plot_df['Locus'], plot_df['F'], color=StyleConfig.WARNING_COLOR, alpha=0.7)
        axes[1,0].set_title('Inbreeding Coefficient (F)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Gráfico 4: PIC
        axes[1,1].bar(plot_df['Locus'], plot_df['PIC'], color=StyleConfig.DANGER_COLOR, alpha=0.7)
        axes[1,1].set_title('Polymorphism Information Content (PIC)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_ssr_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Erro ao criar visualizações: {e}")
        return False

# ====================== FUNÇÕES AFLP (DOMINANTE) ======================

def get_values_map(df, val_range):
    """Mapeia valores AFLP"""
    map_dict = {}
    for col in df.columns:
        df_col = df[col]
        map_dict[col] = []
        for i in df_col:
            if i == 'x':
                map_dict[col] = ['x'] * val_range
            elif not (pd.isnull(i)):
                map_dict[col].append(round(i))
    return map_dict

def get_result_dict(map_dict, min_val, max_val):    
    """Gera dicionário de resultados AFLP"""
    result_dict = {}
    for i in range(min_val, max_val):
        result_dict[i] = {}
        for key, value in map_dict.items():
            if 'x' in value:
                result_dict[i][key] = -9
            else:
                result_dict[i][key] = int(i in value)
    return result_dict

def calculate_aflp_indices(binary_filename):
    """Calcula índices AFLP"""
    try:
        df = pd.read_excel(f'{binary_filename}.xlsx', sheet_name='result', index_col=0)
        
        # Contar todos os loci inicialmente (antes de remover colunas)
        total_loci = df.shape[1]
        
        # Remover colunas com apenas valores -9 (dados faltantes)
        df = df.replace(-9, pd.NA).dropna(axis=1, how='any')
        
        # Contar loci polimórficos após limpeza
        polymorphic_loci = 0
        total_diversity = 0

        for column in df.columns:
            frequency_zero = (df[column] == 0).mean()
            q = frequency_zero ** 0.5
            p = 1 - q
            h = 2 * p * q
            if 0 < frequency_zero < 1:
                polymorphic_loci += 1
                total_diversity += h

        polymorphic_percentage = (polymorphic_loci / total_loci) * 100
        mean_diversity = total_diversity / polymorphic_loci if polymorphic_loci > 0 else 0

        # Salvar arquivo TXT
        with open(f"{binary_filename}_aflp_genetic_indices.txt", "w") as f:
            f.write("AFLP Genetic Indices Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total number of loci: {total_loci}\n")
            f.write(f"Polymorphic loci: {polymorphic_loci}\n")
            f.write(f"Percentage of polymorphic loci (%P): {polymorphic_percentage:.2f}%\n")
            f.write(f"Mean genetic diversity (h): {mean_diversity:.4f}\n")

        # Salvar arquivo PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Genetic Indices - AFLP Markers", ln=True, align="C")
        pdf.set_font("Arial", '', 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Total number of loci: {total_loci}", ln=True)
        pdf.cell(0, 10, f"Polymorphic loci: {polymorphic_loci}", ln=True)
        pdf.cell(0, 10, f"Percentage of polymorphic loci (%P): {polymorphic_percentage:.2f}%", ln=True)
        pdf.cell(0, 10, f"Mean genetic diversity (h): {mean_diversity:.4f}", ln=True)
        pdf.output(f"{binary_filename}_aflp_genetic_indices.pdf")

        return total_loci, polymorphic_loci, polymorphic_percentage, mean_diversity

    except Exception as e:
        raise Exception(f"Error calculating AFLP indices: {e}")

def process_dominant_file(input_file, header_idx, min_val, max_val, output_file):
    """Processa arquivo AFLP"""
    try:
        dfs = pd.read_excel(f'{input_file}.xlsx', sheet_name=None, header=header_idx)
        val_range = max_val - min_val
        current_n_cols = 0

        with pd.ExcelWriter(f'{output_file}.xlsx', engine='openpyxl') as writer:
            for idx, sheet_name in enumerate(dfs):
                df = dfs[sheet_name]
                map_dict = get_values_map(df, val_range)
                result_dict = get_result_dict(map_dict, min_val, max_val)
                result_df = pd.DataFrame(result_dict)
                
                # Remover apenas colunas que somam zero (não contribuem)
                result_df = result_df.drop(columns=result_df.sum()[result_df.sum() <= 0].index)
                
                is_first_df = idx == 0
                result_df.to_excel(writer, sheet_name='result', index=is_first_df, startcol=current_n_cols)
                current_n_cols += len(result_df.columns)
                if is_first_df:
                    current_n_cols += 1

        total, polymorphic, percent, diversity = calculate_aflp_indices(output_file)
        return True, "AFLP analysis completed successfully!", total, polymorphic, percent, diversity
    except Exception as e:
        return False, str(e), None, None, None, None

def create_aflp_visualizations(total_loci, polymorphic_loci, output_prefix):
    """Cria visualizações para dados AFLP"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('AFLP Analysis Summary', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Proporção de loci polimórficos vs monomórficos
        labels = ['Polymorphic', 'Monomorphic']
        sizes = [polymorphic_loci, total_loci - polymorphic_loci]
        colors = [StyleConfig.SUCCESS_COLOR, StyleConfig.SECONDARY_COLOR]
        
        # Verificar se há dados para mostrar
        if sum(sizes) > 0:
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Loci Distribution')
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Loci Distribution - No Data')
        
        # Gráfico 2: Número absoluto de loci
        categories = ['Total Loci', 'Polymorphic Loci']
        values = [total_loci, polymorphic_loci]
        bars = ax2.bar(categories, values, color=[StyleConfig.PRIMARY_COLOR, StyleConfig.SUCCESS_COLOR])
        ax2.set_title('Loci Count')
        ax2.set_ylabel('Number of Loci')
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # Ajustar limite do eixo y para melhor visualização
        max_value = max(values)
        ax2.set_ylim(0, max_value * 1.2)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_aflp_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Erro ao criar visualizações AFLP: {e}")
        return False

# ====================== APLICAÇÃO PRINCIPAL ======================

class ModernGeneticsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ABIgenotyping - Molecular Markers Standardization & First View")
        self.geometry("1000x700")
        self.configure(bg=StyleConfig.LIGHT_BG)
        
        StyleConfig.configure_styles()
        self.setup_ui()
        
    def setup_ui(self):
        """Configura a interface do usuário"""
        # Header
        header_frame = tk.Frame(self, bg=StyleConfig.PRIMARY_COLOR, height=80)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="ABIgenotyping", 
                              font=(StyleConfig.FONT_FAMILY, 24, 'bold'),
                              foreground='white',
                              background=StyleConfig.PRIMARY_COLOR)
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame,
                                 text="Molecular Markers Standardization & First View",
                                 font=(StyleConfig.FONT_FAMILY, 12),
                                 foreground='white',
                                 background=StyleConfig.PRIMARY_COLOR)
        subtitle_label.pack(expand=True)
        
        # Main notebook
        self.main_notebook = ttk.Notebook(self)
        self.main_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Criar abas
        self.create_ssr_tab()
        self.create_aflp_tab()
        
    def create_ssr_tab(self):
        """Cria a aba para análise SSR"""
        ssr_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(ssr_frame, text="🎯 SSR Analysis")
        
        # Sub-notebook para SSR
        self.ssr_notebook = ttk.Notebook(ssr_frame)
        self.ssr_notebook.pack(fill='both', expand=True)
        
        # Aba de processamento
        process_frame = ttk.Frame(self.ssr_notebook)
        self.ssr_notebook.add(process_frame, text="Data Processing")
        
        # Aba de resultados
        results_frame = ttk.Frame(self.ssr_notebook)
        self.ssr_notebook.add(results_frame, text="Genetic Indices")
        
        self.build_ssr_process_tab(process_frame)
        self.build_ssr_results_tab(results_frame)
        
    def build_ssr_process_tab(self, parent):
        """Constrói a interface de processamento SSR"""
        # Container principal
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(main_frame, 
                text="SSR Marker Analysis",
                font=(StyleConfig.FONT_FAMILY, 16, 'bold'),
                foreground=StyleConfig.PRIMARY_COLOR).grid(row=0, column=0, columnspan=3, pady=20)
        
        # Campos de entrada
        fields = [
            ("Excel File:", "ssr_file_entry"),
            ("Sheet Name:", "ssr_sheet_entry"),
            ("Header Row (e.g., 1):", "ssr_header_entry"),
            ("Output Name:", "ssr_output_entry")
        ]
        
        for i, (label, var_name) in enumerate(fields, 1):
            tk.Label(main_frame, text=label, font=(StyleConfig.FONT_FAMILY, 10)).grid(row=i, column=0, sticky='w', padx=10, pady=10)
            entry = ttk.Entry(main_frame, width=50, font=(StyleConfig.FONT_FAMILY, 9))
            entry.grid(row=i, column=1, padx=10, pady=10)
            setattr(self, var_name, entry)
            
            if i == 1:  # Apenas para o campo de arquivo adicionar botão Browse
                ttk.Button(main_frame, text="Browse", command=self.browse_ssr_file).grid(row=i, column=2, padx=5)
        
        # Botão de processamento
        ttk.Button(main_frame, 
                  text="Process SSR Data", 
                  command=self.process_ssr_data,
                  style='Primary.TButton').grid(row=5, column=1, pady=30)
        
    def build_ssr_results_tab(self, parent):
        """Constrói a aba de resultados SSR"""
        # Frame principal
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Treeview para resultados
        columns = ("Locus", "N", "A", "Ae", "Ho", "He", "F", "PIC")
        self.ssr_tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.ssr_tree.heading(col, text=col)
            self.ssr_tree.column(col, width=80, anchor="center")
        
        # Scrollbar para a treeview
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.ssr_tree.yview)
        self.ssr_tree.configure(yscrollcommand=scrollbar.set)
        
        self.ssr_tree.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        scrollbar.grid(row=0, column=1, sticky='ns', pady=5)
        
        # Botão de exportação
        export_frame = ttk.Frame(main_frame)
        export_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=10)
        
        ttk.Button(export_frame, text="Export Results", command=self.export_ssr_results).pack(side='right', padx=5)
        
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
    def create_aflp_tab(self):
        """Cria a aba para análise AFLP"""
        aflp_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(aflp_frame, text="🔬 AFLP Analysis")
        
        # Sub-notebook para AFLP
        self.aflp_notebook = ttk.Notebook(aflp_frame)
        self.aflp_notebook.pack(fill='both', expand=True)
        
        # Aba de processamento
        process_frame = ttk.Frame(self.aflp_notebook)
        self.aflp_notebook.add(process_frame, text="Data Processing")
        
        # Aba de resultados
        results_frame = ttk.Frame(self.aflp_notebook)
        self.aflp_notebook.add(results_frame, text="Analysis Results")
        
        self.build_aflp_process_tab(process_frame)
        self.build_aflp_results_tab(results_frame)
        
    def build_aflp_process_tab(self, parent):
        """Constrói a interface de processamento AFLP"""
        # Container principal
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(main_frame, 
                text="AFLP Marker Analysis",
                font=(StyleConfig.FONT_FAMILY, 16, 'bold'),
                foreground=StyleConfig.PRIMARY_COLOR).grid(row=0, column=0, columnspan=3, pady=20)
        
        # Campos de entrada
        fields = [
            ("Excel File (no extension):", "aflp_file_entry"),
            ("Data Start Row:", "aflp_header_entry"),
            ("Minimum Value to Search:", "aflp_min_val_entry"),
            ("Maximum Value to Search:", "aflp_max_val_entry"),
            ("Output Name:", "aflp_output_entry")
        ]
        
        for i, (label, var_name) in enumerate(fields, 1):
            tk.Label(main_frame, text=label, font=(StyleConfig.FONT_FAMILY, 10)).grid(row=i, column=0, sticky='w', padx=10, pady=8)
            entry = ttk.Entry(main_frame, width=40, font=(StyleConfig.FONT_FAMILY, 9))
            entry.grid(row=i, column=1, padx=10, pady=8)
            setattr(self, var_name, entry)
            
            if i == 1:  # Apenas para o campo de arquivo adicionar botão Browse
                ttk.Button(main_frame, text="Browse", command=self.browse_aflp_file).grid(row=i, column=2, padx=5)
        
        # Botão de processamento
        ttk.Button(main_frame, 
                  text="Process AFLP Data", 
                  command=self.process_aflp_data,
                  style='Primary.TButton').grid(row=6, column=1, pady=25)

        
    def build_aflp_results_tab(self, parent):
        """Constrói a aba de resultados AFLP"""
        # Frame principal
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(main_frame, 
                text="AFLP Analysis Results",
                font=(StyleConfig.FONT_FAMILY, 16, 'bold'),
                foreground=StyleConfig.PRIMARY_COLOR).pack(pady=20)
        
        # Resultados
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill='x', padx=50, pady=20)
        
        # Labels para mostrar resultados
        self.aflp_results_labels = {}
        
        results_data = [
            ("Total Loci:", "total_loci"),
            ("Polymorphic Loci:", "polymorphic_loci"),
            ("% Polymorphic Loci:", "polymorphic_percent"),
            ("Genetic Diversity (h):", "diversity")
        ]
        
        for i, (label, key) in enumerate(results_data):
            tk.Label(results_frame, text=label, font=(StyleConfig.FONT_FAMILY, 11, 'bold')).grid(row=i, column=0, sticky='w', pady=8)
            value_label = tk.Label(results_frame, text="--", font=(StyleConfig.FONT_FAMILY, 11))
            value_label.grid(row=i, column=1, sticky='w', pady=8, padx=10)
            self.aflp_results_labels[key] = value_label
        
    # ====================== MÉTODOS SSR ======================
    
    def browse_ssr_file(self):
        """Abre diálogo para selecionar arquivo SSR"""
        file_path = filedialog.askopenfilename(
            title="Select Excel File for SSR Analysis",
            filetypes=[("Excel Files", "*.xlsx *.xls")]
        )
        if file_path:
            self.ssr_file_entry.delete(0, tk.END)
            self.ssr_file_entry.insert(0, file_path)
            
            # Preencher automaticamente o nome de saída
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if hasattr(self, 'ssr_output_entry'):
                self.ssr_output_entry.delete(0, tk.END)
                self.ssr_output_entry.insert(0, f"{base_name}_processed")
    
    def process_ssr_data(self):
        """Processa dados SSR"""
        try:
            file_path = self.ssr_file_entry.get()
            sheet = self.ssr_sheet_entry.get()
            header_line = int(self.ssr_header_entry.get()) - 1 if self.ssr_header_entry.get() else 0
            output_name = self.ssr_output_entry.get()
            
            if not file_path or not output_name:
                messagebox.showerror("Error", "Please fill in all required fields.")
                return
            
            # Ler e validar dados
            df = pd.read_excel(file_path, sheet_name=sheet, header=header_line)
            
            # Validar dados antes do processamento
            validation_issues = validate_ssr_data(df)
            if validation_issues:
                issues_text = "\n".join(validation_issues)
                messagebox.showwarning("Data Validation", 
                                     f"Foram encontrados os seguintes problemas nos dados:\n\n{issues_text}\n\nAmostras com apenas um alelo serão tratadas como homozigotas.")
            
            # Processar dados
            df_processed = process_codominant_data(df)
            df_processed.to_excel(f"{output_name}.xlsx", index=False)
            
            # Calcular índices
            indices_df, summary_df = calculate_ssr_indices(df_processed)
            final_df = pd.concat([indices_df, summary_df], ignore_index=True)
            
            # Atualizar treeview
            self.update_ssr_treeview(final_df)
            
            # Salvar resultados
            save_ssr_indices_txt(final_df, f"{output_name}_ssr_indices.txt")
            save_ssr_indices_pdf(final_df, f"{output_name}_ssr_indices.pdf")
            create_ssr_visualizations(final_df, output_name)
            
            messagebox.showinfo("Success", 
                              f"SSR analysis completed!\n\n"
                              f"Files saved:\n"
                              f"- {output_name}.xlsx (processed data)\n"
                              f"- {output_name}_ssr_indices.txt\n"
                              f"- {output_name}_ssr_indices.pdf\n"
                              f"- {output_name}_ssr_plots.png")
            
            # Mudar para aba de resultados
            self.ssr_notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing SSR data:\n{str(e)}")
    
    def update_ssr_treeview(self, df):
        """Atualiza a treeview com resultados SSR"""
        # Limpar treeview existente
        for item in self.ssr_tree.get_children():
            self.ssr_tree.delete(item)
        
        # Adicionar novos dados
        for _, row in df.iterrows():
            self.ssr_tree.insert("", "end", values=list(row))
    
    def export_ssr_results(self):
        """Exporta resultados SSR"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if file_path:
                # Aqui você implementaria a exportação dos dados
                messagebox.showinfo("Export", "Export functionality will be implemented")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")
    
    # ====================== MÉTODOS AFLP ======================
    
    def browse_aflp_file(self):
        """Abre diálogo para selecionar arquivo AFLP"""
        file_path = filedialog.askopenfilename(
            title="Select Excel File for AFLP Analysis",
            filetypes=[("Excel Files", "*.xlsx *.xls")]
        )
        if file_path:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.aflp_file_entry.delete(0, tk.END)
            self.aflp_file_entry.insert(0, file_name)
    
    def process_aflp_data(self):
        """Processa dados AFLP"""
        try:
            input_file = self.aflp_file_entry.get().strip()
            header_str = self.aflp_header_entry.get().strip()
            min_val_str = self.aflp_min_val_entry.get().strip()
            max_val_str = self.aflp_max_val_entry.get().strip()
            output_file = self.aflp_output_entry.get().strip()
            
            if not all([input_file, header_str, min_val_str, max_val_str, output_file]):
                messagebox.showerror("Error", "Please fill in all fields.")
                return
            
            # Converter valores
            header_idx = int(header_str) - 1
            min_val = int(min_val_str)
            max_val = int(max_val_str) + 1  # +1 para incluir o valor máximo
            
            # Processar arquivo
            success, msg, total, polymorphic, percent, diversity = process_dominant_file(
                input_file, header_idx, min_val, max_val, output_file
            )
            
            if success:
                # Atualizar interface com resultados
                self.aflp_results_labels["total_loci"].config(text=str(total))
                self.aflp_results_labels["polymorphic_loci"].config(text=str(polymorphic))
                self.aflp_results_labels["polymorphic_percent"].config(text=f"{percent:.2f}%")
                self.aflp_results_labels["diversity"].config(text=f"{diversity:.4f}")
                
                # Criar visualizações
                create_aflp_visualizations(total, polymorphic, output_file)
                
                messagebox.showinfo("Success", 
                                  f"AFLP analysis completed!\n\n"
                                  f"Files saved:\n"
                                  f"- {output_file}.xlsx (binary matrix)\n"
                                  f"- {output_file}_aflp_genetic_indices.txt\n"
                                  f"- {output_file}_aflp_genetic_indices.pdf\n"
                                  f"- {output_file}_aflp_plots.png")
                
                # Mudar para aba de resultados
                self.aflp_notebook.select(1)
            else:
                messagebox.showerror("Error", f"AFLP processing failed:\n{msg}")
                
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values for header row and value ranges.")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error processing AFLP data:\n{str(e)}")

def main():
    """Função principal"""
    try:
        app = ModernGeneticsApp()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to start application:\n{str(e)}")

if __name__ == "__main__":
    main()
