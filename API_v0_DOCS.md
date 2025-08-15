# Face Finder API v0 Documentation

## Overview

The Face Finder API v0 provides two main endpoints for face recognition tasks:

1. **`/api/v0/embed`** - Pre-warm the cache by downloading and creating face embeddings for multiple images
2. **`/api/v0/findIn`** - Find a target person in a set of scope images

## Endpoints

### POST /api/v0/embed

Pre-warm the cache by downloading images and creating face embeddings. Each embedding is saved in a separate file named by the hash of the URL.

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

Find a target person in a set of scope images. This endpoint will use cached embeddings if available, or compute them on the fly.

**Request Body:**

```json
{
  "target": "https://example.com/person.jpg",
  "scope": ["https://example.com/group1.jpg", "https://example.com/group2.jpg", "https://example.com/group3.jpg"],
  "threshold": 0.6
}
```

**Parameters:**

- `target` (string, required): URL of the image containing the person to find
- `scope` (array, required): Array of URLs to search for the target person
- `threshold` (float, optional): Similarity threshold for matching (0.0 to 1.0, default: 0.6)

**Response:**

```json
{
  "success": true,
  "target_url": "https://example.com/person.jpg",
  "threshold": 0.6,
  "total_scope_images": 3,
  "total_matches": 2,
  "matches": [
    {
      "url": "https://example.com/group1.jpg",
      "similarity": 0.8521,
      "matching_faces": 1,
      "all_similarities": [0.8521]
    },
    {
      "url": "https://example.com/group3.jpg",
      "similarity": 0.7234,
      "matching_faces": 1,
      "all_similarities": [0.7234]
    }
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
- Cache files are stored in `cache/embeddings/temp_reference_{url_hash}_embeddings.pkl`
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
