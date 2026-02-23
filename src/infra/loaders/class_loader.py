import importlib

def build_class(cls_path, args=None):
    cls = load_class(cls_path)
    return cls(**(args or {}))

def load_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)