def safe_float_format(value, precision=4):
    """Безопасное форматирование чисел с плавающей точкой"""
    if value is None:
        return "N/A"
    try:
        return f"{value:.{precision}f}"
    except (TypeError, ValueError):
        return "N/A"
