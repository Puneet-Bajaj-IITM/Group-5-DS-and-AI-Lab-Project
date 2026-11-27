# AniDex assets

The AniDex tab names common animals: pets, farm animals, everyday birds and well-known wildlife, 154 species in all. For the 17 domestic animals that have breeds (dog, cat, cattle, water buffalo, sheep, pig, horse, donkey, goat, chicken, duck, goose, turkey, pigeon, rabbit, guinea pig and goldfish), it also names the breed, out of 335 breeds. Wild animals have no breeds; their cards list the subspecies GBIF knows instead. The AniDex does not feed a snake verdict.

| File | In git | What it is |
|---|---|---|
| `anidex.json` | yes | Species, breeds, which label belongs to which species, preprocessing settings, and the card text (Wikipedia summaries, GBIF taxonomy, IUCN status), so the cards work offline |
| `anidex_labels.bin` | yes | BioCLIP 2's label vectors for the species step: float32, row-major, `label_count x dim` |
| `anidex_breed_labels.bin` | yes | The general CLIP's breed vectors: float32, row-major, `breed_label_count x breed_dim` |
| `anidex_encoder.onnx` | no (`*.onnx` is ignored) | BioCLIP 2's image encoder, int8 weights, about 308 MB |
| `anidex_breed_encoder.onnx` | no | The general CLIP's image encoder (ViT-B/16 trained on DataComp), int8 weights |

Both encoders take the same 1x3x224x224 input (`image`) and return an L2-normalised embedding (`embedding`). Rebuild everything from `app/`, using Python with `pybioclip`, `onnx`, `onnxruntime` and `onnxconverter-common`:

```bash
python tool/anidex/build_anidex.py
```

The first run downloads both models (about 5 GB including BioCLIP's Tree of Life label embeddings). Options:
- `--skip-cards` reuses the card text already in `anidex.json` and fetches only what is missing.
- `--skip-encoder` keeps the exported encoders.
- `--check-photos DIR` compares the ONNX encoders with PyTorch on a folder of photos.

To add an animal or a breed, edit `tool/anidex/animals.py` and rebuild.

## How it works

Both models are used zero-shot: nothing is trained on these animals. The text side is computed once, here, so the phone runs only the two image encoders.

- **Species:** [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) (Imageomics Institute, MIT licence). Each species has several labels: its Tree of Life text embedding, a plain-English prompt ("a photo of a cow"), and one prompt per breed. The phone computes `softmax(100 · labels · embedding)` and adds up each species' label scores. Below 50%, the card warns that the photo may not show one of the 154 animals, and the scan is not added to the collection.
- **Breed:** a general CLIP (OpenCLIP ViT-B/16, DataComp-1B). It runs only when the species has breeds, and only that species' breeds are scored. Below 40%, the card says "Mixed or unknown breed" and shows the top three guesses.

**Why two models:** BioCLIP 2 learned from taxonomic names, which stop at species, so it barely knows breeds. On 28 breed photos it named the right breed 6 or 7 times. The general CLIP, which learned from web captions, got 19 of 28. Its misses are mostly real lookalikes (Jersey vs Guernsey, Holland Lop vs Mini Lop, Large White vs Landrace), and Indian breeds are its weakest area. Combining the two scores helped little (20 of 28), so the breed comes from the general CLIP alone.

**Accuracy:** on 43 photos, BioCLIP 2 named every species correctly. That includes 21 everyday photos (dogs, cats, cows, a buffalo, a goat, a horse, birds, a rabbit and some wild animals) and 22 breed photos.

**Preprocessing:** both encoders get the photo stretched to 224 x 224 with the same antialiased bilinear filter torchvision uses (`preprocess` in `lib/anidex/anidex_model.dart`). The general CLIP normally centre-crops with a bicubic filter, but on the breed photos the stretched input scored the same (19 vs 18 of 28), so the phone prepares the photo once for both models. An earlier version used `package:image`'s resize, which was 0.19 off on average, and that was enough to turn a Siamese cat into a dog. `test/anidex_test.dart` checks the Dart preprocessing and scoring against fixtures from `tool/anidex/make_fixtures.py`.

## Size and speed

The two encoders make the app about 380 MB bigger (294 MB and 85 MB): the arm64 debug APK is 534 MB. The first scan copies both encoders out of the app once, which took about 5 s on an Android emulator. After that, the species model takes about 0.2 s per photo and the breed model about 0.05 s.

On the emulator, the app named Golden Retriever, Holstein Friesian, Labrador Retriever and Siamese cat correctly. A Persian cat came out as Exotic Shorthair, the short-haired Persian, which is a close lookalike.

**Breed descriptions:** 14 breeds (some Indian sheep, goats and donkeys, two guinea pig breeds, Nicobari chicken and LaMancha) have no English Wikipedia article. Their cards show the animal's own description instead. The smaller BioCLIP (v1, ViT-B/16) was tested for the species step and rejected: it got 2 of 21 species wrong (a Siamese cat came out as a kangaroo). If an encoder is missing from the build, the AniDex tab says so. If `anidex.json` has `encoder_url` and `breed_encoder_url`, the app downloads the encoders from there on first use instead, and checks their SHA-256.
