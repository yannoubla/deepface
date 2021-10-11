from deepface import DeepFace
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("db_path", help="database path of reference images")
parser.add_argument("avatar_path", help="path of avatars")
args = parser.parse_args()

DeepFace.stream(db_path=args.db_path, avatar_path=args.avatar_path, enable_face_analysis=True)