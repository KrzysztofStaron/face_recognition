<?php
$ch = curl_init('http://75.119.154.215:5003/api/v0/embed');
$payload = json_encode([
  'urls' => [
    'https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file=demowki085.jpg'
  ]
]);

curl_setopt_array($ch, [
  CURLOPT_POST => true,
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_HTTPHEADER => ['Content-Type: application/json', 'Accept: application/json'],
  CURLOPT_POSTFIELDS => $payload,
  CURLOPT_TIMEOUT => 60,
]);

$response = curl_exec($ch);
if ($response === false) {
  throw new Exception('cURL error: ' . curl_error($ch));
}

$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($httpCode >= 400) {
  throw new Exception("HTTP $httpCode: $response");
}

$data = json_decode($response, true);
print_r($data);