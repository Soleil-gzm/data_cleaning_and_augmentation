import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist if 'WenQuanYi' in f.name]
print(fonts)