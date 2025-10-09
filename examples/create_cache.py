import sys
from pathlib import Path
from datasets.colmap import Parser 

sys.path.append('/lustre/fsw/portfolios/nvr/users/ymingli/projects/gsplat-city/submodules/gsplat') 

def generate_parser_cache(data_dir):
    print(f"Starting cache generation for: {data_dir}")
    print("This may take a long time...")

    _ = Parser(
        data_dir=data_dir,
        factor=1,
        normalize=False,
        test_every=60,
    )

    cache_file = Path(data_dir) / "parser_cache.pkl"
    if cache_file.exists():
        print(f"✅ Successfully created cache file at: {cache_file}")
    else:
        print("❌ Error: Cache file was not created.")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python create_cache.py <path_to_data_dir>")
    else:
        data_directory = sys.argv[1]
        generate_parser_cache(data_directory)