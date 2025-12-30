import matplotlib.font_manager as fm

fonts = sorted([a.name for a in fm.fontManager.ttflist])
print(fonts)