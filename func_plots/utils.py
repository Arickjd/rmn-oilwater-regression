"""Exportação de figuras raster e vetoriais para o artigo."""

from pathlib import Path


def save_figure(figure, path, formats=("png", "pdf"), dpi=300):
    """Salva nos formatos pedidos e retorna os caminhos; path é o nome-base."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        destination = path.with_suffix(f".{extension}")
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        paths.append(destination)
    return paths

