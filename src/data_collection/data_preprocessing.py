def handle_missing_values(data):
    data.fillna(0, inplace=True)
    return data


