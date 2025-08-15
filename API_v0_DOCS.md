# Face Finder API v0 Documentation

## Overview

The Face Finder API v0 provides these endpoints for face recognition tasks:

1. **`/api/v0/embed`** - Pre-warm the cache by downloading and creating face embeddings for multiple images
2. **`/api/v0/findIn`** - Find a target person in a set of scope images
3. **`/api/v0/inspect`** - Inspect an image to list detected faces (index, bbox, score)

## Endpoints

### POST /api/v0/embed

Pre-warm the cache by downloading images and creating face embeddings. Each embedding is saved in a separate file named by the hash of the URL. A companion faces metadata file (with bboxes/scores) is also created.

**Request Body:**

```json
{
  "urls": ["https://example.com/image1.jpg", "https://example.com/image2.jpg", "https://example.com/image3.jpg"]
}
```

**Response:**

```json
{
  "success": true,
  "total_urls": 3,
  "results": [
    {
      "url": "https://example.com/image1.jpg",
      "success": true,
      "cached": true,
      "num_faces": 1,
      "cache_file": "cache/embeddings/temp_reference_abc123_embeddings.pkl"
    },
    {
      "url": "https://example.com/image2.jpg",
      "success": true,
      "cached": true,
      "num_faces": 2,
      "cache_file": "cache/embeddings/temp_reference_def456_embeddings.pkl"
    }
  ]
}
```

### POST /api/v0/findIn

Find a target person in a set of scope images. If the target image contains multiple faces, the endpoint can search for all faces, the largest/best one, or a specific face index. The endpoint performs greedy one-to-one matching to avoid duplicate counting. It uses cached embeddings if available, or computes them on the fly. You can optionally include face details (bboxes, detection scores) in the response.

**Request Body:**

```json
{
  "target": "https://example.com/person.jpg",
  "scope": ["https://example.com/group1.jpg", "https://example.com/group2.jpg", "https://example.com/group3.jpg"],
  "threshold": 0.6,
  "target_face": "all", // optional: "all" (default) | "largest" | "best" | index | [indices]
  "include_details": false, // optional: include bboxes/scores
  "max_results": 100 // optional: limit result count
}
```

Notes:

- `target_face` can be one of: "all" (default), "largest", "best", a number index (0-based), or an array of indices. Negative indices allowed.
- `include_details: true` adds bounding boxes and detection scores to each detailed match and returns `selected_target_indices` and `target_summary` at the top level.

**Parameters:**

- `target` (string, required): URL of the image containing the person to find
- `scope` (array, required): Array of URLs to search for the target person
- `threshold` (float, optional): Similarity threshold for matching (0.0 to 1.0, default: 0.6)
- `target_face` (optional): Which target face(s) to search with. One of:
  - `"all"` (default): use all faces in the target image
  - `"largest"`: face with the largest bounding box
  - `"best"`: face with the highest detection score
  - `number` or `number[]`: select by index(es) (0-based; negative indices allowed)
- `include_details` (boolean, optional): Whether to include face bboxes/scores in `detailed_matches`
- `max_results` (number, optional): Limit the number of returned matches

**Response:**

```json
{
  "success": true,
  "target_url": "https://example.com/person.jpg",
  "target_faces_count": 2,
  "threshold": 0.6,
  "total_scope_images": 3,
  "total_matches": 2,
  "urls": ["1.jpg", "2.jpg", "3.jpg"],
  "matches": [
    {
      "url": "https://example.com/group1.jpg",
      "similarity": 0.8521,
      "target_faces_found": 2,
      "target_face_indices": [0, 1]
    }
  ]
}
```

**Response Fields:**

- `target_faces_count`: Number of faces detected in the target image
- `target_faces_found`: Number of target faces found in this scope image
- `target_face_indices`: Array of target face indices that were found
- If `include_details` is true: `face_matches` is included per image with entries containing `target_face`, `scope_face`, `similarity`, and optionally `target_bbox`, `target_score`, `scope_bbox`, `scope_score`.
- `urls`: Array of scope image URLs where the target face(s) were detected (most important for consumers)
- If `include_details` is true (top-level):
  - `selected_target_indices`: Which target faces were selected for matching
  - `target_summary`: For each detected target face, include `index`, `bbox`, `score`

### POST /api/v0/inspect

Inspect an image to list detected faces and their metadata so the client can choose a specific face index for subsequent calls.

**Request Body:**

```
{ "url": "https://example.com/image.jpg" }
```

**Response:**

```
{
  "success": true,
  "url": "https://example.com/image.jpg",
  "faces_count": 2,
  "faces": [
    { "index": 0, "bbox": [x1, y1, x2, y2], "score": 0.998 },
    { "index": 1, "bbox": [x1, y1, x2, y2], "score": 0.992 }
  ]
}
```

## Typical Workflow

1. **Pre-warm cache** (optional but recommended for better performance):

   ```bash
   curl -X POST http://localhost:5003/api/v0/embed \
     -H "Content-Type: application/json" \
     -d '{"urls": ["url1", "url2", "url3"]}'
   ```

2. **Find person in images**:
   ```bash
   curl -X POST http://localhost:5003/api/v0/findIn \
     -H "Content-Type: application/json" \
     -d '{
       "target": "https://example.com/person.jpg",
       "scope": ["url1", "url2", "url3"],
       "threshold": 0.6
     }'
   ```

## Caching

- Embeddings are cached using a hash of the URL as the filename
- Embeddings cache file: `cache/embeddings/temp_reference_{url_hash}_embeddings.pkl`
- Faces metadata cache file (bbox/score/landmarks + embedding): `cache/embeddings/temp_reference_{url_hash}_faces.pkl`
- Once an image is cached, subsequent requests will use the cached embeddings
- The cache persists across server restarts

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK` - Successful request
- `400 Bad Request` - Invalid request parameters
- `500 Internal Server Error` - Server error

Error responses include a descriptive message:

```json
{
  "success": false,
  "error": "Description of the error"
}
```

## Performance Tips

1. Use `/api/v0/embed` to pre-warm the cache before calling `/api/v0/findIn`
2. Cached embeddings significantly improve response times
3. Higher thresholds (e.g., 0.7-0.8) give more accurate matches but fewer results
4. Lower thresholds (e.g., 0.4-0.5) give more results but may include false positives
