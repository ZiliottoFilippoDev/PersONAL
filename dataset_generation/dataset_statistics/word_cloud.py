"""
This is the file to produce some wordcloud images
"""

import os
import argparse
from dataset_statistics.utils import load_scenes_objects
from dataset_statistics.utils import extract_descriptions, extract_objects, extract_object_room
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter

def generate_wordcloud(descriptions, 
                       output_dir='dataset_statistics/images', 
                       output_filename='wordcloud.png', 
                       **wordcloud_kwargs):
    """
    Generate a word cloud from a list of description strings and save it as an image.

    Parameters
    ----------
    descriptions : list of str
        A list of textual descriptions.
    output_dir : str, optional
        Directory path where the image will be saved. Default is 'dataset_statistics/images'.
    output_filename : str, optional
        Name of the output image file. Default is 'wordcloud.png'.
    **wordcloud_kwargs :
        Additional keyword arguments to pass to WordCloud, e.g., max_words, background_color, width, height.

    Returns
    -------
    output_path : str
        Full path to the saved word cloud image.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all descriptions into one large text
    text = ' '.join(descriptions)
    
    # Default parameters for WordCloud if not provided
    defaults = {
        'width': 800,
        'height': 400,
        'background_color': 'white',
        'max_words': 200
    }
    # Merge defaults with any user-supplied overrides
    defaults.update(wordcloud_kwargs)
    
    # Create the word cloud object
    wc = WordCloud(**defaults)
    wc.generate(text)
    
    # Plot and save
    plt.figure(figsize=(defaults['width']/100, defaults['height']/100))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    
    # Build full output path and save file
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return output_path

def generate_histogram(objects, 
                       output_dir='dataset_statistics/images', 
                       output_filename='histogram.png', 
                       min_count=5,
                       **histogram_kwargs):
    """
    Generate a histogram of object name frequencies from a list of strings, grouping low-frequency items into 'Others', and save it as an image.

    Parameters
    ----------
    objects : list of str
        A list of object names.
    output_dir : str, optional
        Directory path where the image will be saved. Default is 'images'.
    output_filename : str, optional
        Name of the output image file. Default is 'histogram.png'.
    min_count : int, optional
        Minimum frequency an object must have to be shown individually. Others are grouped. Default is 5.
    **histogram_kwargs :
        Additional keyword arguments for matplotlib.pyplot.bar, e.g., width.

    Returns
    -------
    output_path : str
        Full path to the saved histogram image.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Count frequencies
    freq = Counter(objects)

    # Separate frequent items and group the rest
    frequent = {name: count for name, count in freq.items() if count >= min_count}
    others_count = sum(count for name, count in freq.items() if count < min_count)
    if others_count > 0:
        frequent['others'] = others_count

    # Sort items descending
    items = sorted(frequent.items(), key=lambda x: x[1], reverse=True)
    names, counts = zip(*items) if items else ([], [])

    # Plot histogram
    plt.figure(figsize=(12, 6))
    bar_kwargs = {'align': 'center'}
    bar_kwargs.update(histogram_kwargs)
    plt.bar(names, counts, **bar_kwargs)
    plt.xlabel('Object Names', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Object Frequency Histogram', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Build full output path and save file
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.close()

    return output_path

def generate_top_tokens_bar_chart(descriptions,
                                  top_n=50,
                                  base_path='dataset_statistics/images',
                                  output_filename='top_tokens.png',
                                  stop_words=None,
                                  **bar_kwargs):
    """
    Generate a bar chart of the most common tokens from a list of descriptions, excluding stop words, and save it as an image.

    Parameters
    ----------
    descriptions : list of str
        A list of textual descriptions.
    top_n : int, optional
        Number of top tokens to display. Default is 50.
    base_path : str, optional
        Directory path where the image will be saved. Default is 'images'.
    output_filename : str, optional
        Name of the output image file. Default is 'top_tokens.png'.
    stop_words : set of str, optional
        Tokens to exclude. Defaults to the WordCloud STOPWORDS set.
    **bar_kwargs :
        Additional keyword arguments for matplotlib.pyplot.barh, e.g., height.

    Returns
    -------
    output_path : str
        Full path to the saved bar chart image.
    """
    os.makedirs(base_path, exist_ok=True)

    # Determine stop words
    stopset = stop_words if stop_words is not None else set(STOPWORDS)
    
    # Addd stuff i see
    stopset.update(["it."])

    # Tokenize, lowercase, and filter stop words
    tokens = [token.lower() for desc in descriptions for token in desc.split()]
    filtered = [token for token in tokens if token not in stopset]

    # Count and take most common
    freq = Counter(filtered)
    most_common = freq.most_common(top_n)

    tokens, counts = zip(*most_common) if most_common else ([], [])

    # Plot horizontal bar chart
    plt.figure(figsize=(10, max(6, top_n * 0.2)))
    defaults = {'align': 'center'}
    defaults.update(bar_kwargs)
    plt.barh(tokens[::-1], counts[::-1], **defaults)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Tokens', fontsize=12)
    plt.title(f'Top {top_n} Tokens by Frequency (stop words removed)', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(base_path, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Calculate dataset statistics.")
    parser.add_argument("--base_path", type=str, default="data/datasets/eai_pers", help="Base path to dataset")
    parser.add_argument("--split", type=str, choices=["total", "divide"], default="total", help="Split type to load") 
    parser.add_argument("--output_dir", type=str, default="dataset_statistics/images", help="Directory to save the images") 
    parser.add_argument("--generate_for_room", action="store_true", default=True, help="Generate word cloud for room names")  
    args = parser.parse_args()
    
    print("=====================================")
    print("Starting to calculate dataset statistics...")
    print("=====================================")
    
    args.split="total"
    if args.split=="total":
        print("Loading the total seen + unseen...")
        # Load the objects from the dataset
        object_list = load_scenes_objects(base_dir=args.base_path, split="total")
        
        # Get a list of only the descriptions for each object
        descriptions = extract_descriptions(object_list)
        # Generate the word cloud
        print("Generating word cloud for total seen + unseen...")
        generate_wordcloud(descriptions, output_dir=args.output_dir,  output_filename='wordcloud_total.png')
        
        print("Generating histogram for total seen + unseen...")
        objects = extract_objects(object_list)
        generate_histogram(objects, output_dir=args.output_dir, output_filename='histogram_total.png')
        
        if args.generate_for_room:
            print("Generating histogram for room names...")
            objects = extract_object_room(object_list)
            generate_wordcloud(objects, output_dir=args.output_dir, output_filename='wordcloud_room_total.png')
            generate_histogram(objects, output_dir=args.output_dir, output_filename='histogram_room_total.png')
        
        # Generate top tokens bar chart
        print("Generating top tokens bar chart...")
        generate_top_tokens_bar_chart(descriptions, top_n=50, base_path=args.output_dir, output_filename='top_tokens_total.png')
        
    elif args.split=="divide":
        print("Subdiving the dataset into seen and unseen...")
        val_seen_objects, val_unseen_objects = load_scenes_objects(split="divide")
    
        # Get a list of only the descriptions for each object
        descriptions_seen = extract_descriptions(val_seen_objects)
        # Generate the word cloud
        print("Generating word cloud for seen objects...")
        generate_wordcloud(descriptions_seen, output_dir=args.output_dir, output_filename='wordcloud_seen.png')
        print("Generating histogram for seen objects...")
        objects_seen = extract_objects(val_seen_objects)
        generate_histogram(objects_seen, output_dir=args.output_dir, output_filename='histogram_seen.png', min_count=3)
        # Get a list of only the descriptions for each object
        descriptions_unseen = extract_descriptions(val_unseen_objects)
        # Generate the word cloud
        print("Generating word cloud for unseen objects...")
        generate_wordcloud(descriptions_unseen, output_dir=args.output_dir, output_filename='wordcloud_unseen.png')
        print("Generating histogram for unseen objects...")
        objects_unseen = extract_objects(val_unseen_objects)
        generate_histogram(objects_unseen, output_dir=args.output_dir, output_filename='histogram_unseen.png', min_count=3)
    
        # Generate top tokens bar chart
        print("Generating top tokens bar chart for seen objects...")
        generate_top_tokens_bar_chart(descriptions_seen, top_n=50, base_path=args.output_dir, output_filename='top_tokens_seen.png')
        print("Generating top tokens bar chart for unseen objects...")
        generate_top_tokens_bar_chart(descriptions_unseen, top_n=50, base_path=args.output_dir, output_filename='top_tokens_unseen.png')
    
        if args.generate_for_room:
            print("Generating histogram for room names...")
            objects_seen = extract_object_room(val_seen_objects)
            generate_wordcloud(objects_seen, output_dir=args.output_dir, output_filename='wordcloud_room_seen.png')
            generate_histogram(objects_seen, output_dir=args.output_dir, output_filename='histogram_room_seen.png', min_count=2)
            objects_unseen = extract_object_room(val_unseen_objects)
            generate_wordcloud(objects_unseen, output_dir=args.output_dir, output_filename='wordcloud_room_unseen.png')
            generate_histogram(objects_unseen, output_dir=args.output_dir, output_filename='histogram_room_unseen.png', min_count=2)
    
    print("=========")
    print("Finished")
    print("=========")