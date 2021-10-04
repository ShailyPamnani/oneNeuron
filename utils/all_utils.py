  def prepare_data(df):
    X = df.drop("y", axis=1)

    y = df["y"]

    return X, y