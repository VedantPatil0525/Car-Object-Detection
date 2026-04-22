# 🚗 Running the Streamlit UI

## Prerequisites
Make sure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

## Start the UI

Run the Streamlit app:
```bash
streamlit run app.py
```

This will start a local web server, typically at `http://localhost:8501`

## Features

### 1. **Predict** 🎯
- Upload a single image (JPG, PNG, BMP)
- Select which model to use (Adam, SGD, or RMSprop)
- View the predicted bounding box on the image
- See exact coordinate values (xmin, ymin, xmax, ymax)

### 2. **Compare Models** 🔄
- Upload an image once
- See predictions from all 3 models side-by-side
- Compare predicted coordinates across optimizers
- View results in a detailed comparison table

### 3. **Training Metrics** 📊
- View loss curves for all three optimizers
- Compare training vs validation loss
- See final metrics for each model
- Analyze which optimizer performed best

### 4. **About** ℹ️
- Project overview and architecture
- Technology stack information
- Quick reference guide

## Supported Image Formats
- JPG / JPEG
- PNG
- BMP

## Tips
- Images are automatically resized to 224×224 for the model
- The UI uses cached models for faster predictions
- All three models (adam, sgd, rmsprop) should be in the `models/` directory

## Troubleshooting

**Model not found error?**
- Make sure you've trained the models by running `python main.py`
- Check that model files exist in `models/` directory

**Slow predictions?**
- First prediction loads the model into memory (slower)
- Subsequent predictions are faster due to Streamlit caching

**Can't upload images?**
- Check file format (JPG, PNG, BMP supported)
- File size should be reasonable (< 100MB)
