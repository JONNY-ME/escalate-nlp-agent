import matplotlib.pyplot as plt

def plot_lengths(length_df, outpath):
    ax = length_df["length"].hist(bins=50)
    ax.set_xlabel("Tokens per document")
    ax.set_ylabel("Count")
    ax.set_title("Document Lengths")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_top_terms(top_df, outpath):
    ax = top_df.set_index("term")["count"].plot(kind="barh")
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Top Terms")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
