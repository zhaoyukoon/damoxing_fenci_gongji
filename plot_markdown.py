import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

def get_styles(style_count):
    """
    Generate a list of distinct plot styles.
    """
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 
             'cyan', 'magenta', 'gray', 'olive', 'pink', 'teal']
    
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x', '|', '_']
    
    styles = []
    for i in range(style_count):
        style = {
            'color': colors[i % len(colors)],
            'linestyle': linestyles[(i // len(colors)) % len(linestyles)],
            'marker': markers[i % len(markers)]
        }
        styles.append(style)
    
    return styles

def plot_length_distribution(lengths_pairs, vocab_names):
    """
    Plot length distribution for multiple vocabularies.
    """
    plt.figure(figsize=(10, 6))
    styles = get_styles(2*len(vocab_names))
    
    for i in range(len(vocab_names)):
        all_lens = lengths_pairs[i][0]
        chinese_lens = lengths_pairs[i][1]
        
        all_unique_lengths, all_counts = np.unique(all_lens, return_counts=True)
        chinese_unique_lengths, chinese_counts = np.unique(chinese_lens, return_counts=True)
        
        plt.plot(all_unique_lengths, 
                all_counts,
                color=styles[2*i]['color'],
                linestyle=styles[2*i]['linestyle'],
                marker=styles[2*i]['marker'],
                markersize=4,
                linewidth=1.5,
                label=f"{vocab_names[i]}_all")
        
        plt.plot(chinese_unique_lengths, 
                chinese_counts,
                color=styles[2*i+1]['color'],
                linestyle=styles[2*i+1]['linestyle'],
                marker=styles[2*i+1]['marker'],
                markersize=4,
                linewidth=1.5,
                label=f"{vocab_names[i]}_chinese")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Vocabulary Length Distribution')
    plt.xlabel('Vocabulary Length (log scale)')
    plt.ylabel('Count (log scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('vocab_length_histogram.png', dpi=300, bbox_inches='tight')
    logger.info('Saved histogram to vocab_length_histogram.png')

def write_lang_count_markdown(model_to_lang_count):
    """
    Write language distribution statistics to a markdown file.
    """
    logger.info('Writing language counts to model_lang_count.md')
    
    with open('大模型词汇语言分布.md', 'w', encoding='utf-8') as f:
        f.write('# 大模型词汇语言分布\n\n')
        
        models = sorted(list(model_to_lang_count.keys()))
        languages = set()
        for lang_count in model_to_lang_count.values():
            languages.update(lang_count.keys())
        languages = sorted(list(languages))
        
        model_totals = {}
        for model, lang_count in model_to_lang_count.items():
            model_totals[model] = sum(lang_count.values())
        
        f.write('| Language | ' + ' | '.join(models) + ' |\n')
        f.write('|----------|' + '|'.join(['---' for _ in range(len(models))]) + '|\n')
        
        for lang in languages:
            row = [lang]
            for model in models:
                count = model_to_lang_count[model].get(lang, 0)
                row.append(str(count))
            f.write('| ' + ' | '.join(row) + ' |\n')
        
        total_row = ['Total']
        for model in models:
            total_row.append(str(model_totals[model]))
        f.write('| ' + ' | '.join(total_row) + ' |\n')
