// ESP-NOW link for face24 payloads: the Core2 scanner broadcasts each scanned payload,
// the T-Deck receives, reassembles and decodes it. No pairing: broadcast on a fixed channel.
//
// Packet: 12-byte header + up to 200 payload bytes (ESP-NOW v1 limit is 250 bytes).
//   magic "F24" | version 1 | msg_id (u32) | total_len (u16) | part (u8) | parts (u8) | data
// A 256-byte payload is 2 packets. Each message is sent a few times; the receiver ignores
// packets of a message it has already completed.
#pragma once
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <freertos/FreeRTOS.h>
#include <string.h>

namespace face24link {

static constexpr uint8_t CHANNEL      = 1;
static constexpr size_t CHUNK         = 200;
static constexpr size_t MAX_PAYLOAD   = 1024;
static constexpr int SEND_REPEATS     = 3;
static const uint8_t BROADCAST[6]     = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

struct __attribute__((packed)) Header {
    char magic[3];  // "F24"
    uint8_t version;
    uint32_t msg_id;
    uint16_t total_len;
    uint8_t part;
    uint8_t parts;
};
static_assert(sizeof(Header) == 12, "link header layout");

// Wi-Fi in station mode on CHANNEL, no access point; ESP-NOW initialised.
inline bool begin()
{
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    if (esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE) != ESP_OK) return false;
    return esp_now_init() == ESP_OK;
}

/* ------------------------------------------------------------- sender */

inline bool beginSender()
{
    if (!begin()) return false;
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, BROADCAST, 6);
    peer.channel = CHANNEL;
    peer.ifidx   = WIFI_IF_STA;
    peer.encrypt = false;
    return esp_now_add_peer(&peer) == ESP_OK;
}

// Broadcasts one payload (split into packets), SEND_REPEATS times. Returns false if any send failed.
inline bool send(const uint8_t* payload, size_t len)
{
    if (len == 0 || len > MAX_PAYLOAD) return false;
    uint8_t buf[sizeof(Header) + CHUNK];
    Header h;
    memcpy(h.magic, "F24", 3);
    h.version   = 1;
    h.msg_id    = esp_random();
    h.total_len = (uint16_t)len;
    h.parts     = (uint8_t)((len + CHUNK - 1) / CHUNK);
    bool ok     = true;
    for (int r = 0; r < SEND_REPEATS; r++) {
        for (uint8_t part = 0; part < h.parts; part++) {
            size_t off = (size_t)part * CHUNK, n = min(CHUNK, len - off);
            h.part     = part;
            memcpy(buf, &h, sizeof(h));
            memcpy(buf + sizeof(h), payload + off, n);
            ok &= esp_now_send(BROADCAST, buf, sizeof(h) + n) == ESP_OK;
            delay(4);  // let the radio drain its queue
        }
        delay(30);
    }
    return ok;
}

/* ----------------------------------------------------------- receiver */

// Reassembles messages in the ESP-NOW receive callback (Wi-Fi task); the main loop
// takes complete payloads with take(). Keeps only the newest complete payload.
class Receiver {
   public:
    void onPacket(const uint8_t* data, int len)
    {
        if (len < (int)sizeof(Header)) return;
        Header h;
        memcpy(&h, data, sizeof(h));
        if (memcmp(h.magic, "F24", 3) != 0 || h.version != 1 || h.total_len == 0 || h.total_len > MAX_PAYLOAD ||
            h.parts == 0 || h.parts > 8 || h.part >= h.parts || h.parts != (h.total_len + CHUNK - 1) / CHUNK)
            return;
        if (h.msg_id == done_id_) return;  // a repeat of a message we already have
        if (h.msg_id != cur_id_) {         // start of a new message
            cur_id_   = h.msg_id;
            cur_len_  = h.total_len;
            cur_mask_ = 0;
        }
        size_t off = (size_t)h.part * CHUNK, n = len - sizeof(Header);
        if (n != min(CHUNK, (size_t)cur_len_ - off)) return;
        memcpy(cur_ + off, data + sizeof(Header), n);
        cur_mask_ |= 1u << h.part;
        if (cur_mask_ == (1u << h.parts) - 1) {
            portENTER_CRITICAL(&lock_);
            memcpy(ready_, cur_, cur_len_);
            ready_len_ = cur_len_;
            portEXIT_CRITICAL(&lock_);
            done_id_ = cur_id_;
        }
    }

    // Copies the newest complete payload into out (MAX_PAYLOAD bytes) and returns its length, or 0.
    size_t take(uint8_t* out)
    {
        portENTER_CRITICAL(&lock_);
        size_t n = ready_len_;
        if (n) memcpy(out, ready_, n);
        ready_len_ = 0;
        portEXIT_CRITICAL(&lock_);
        return n;
    }

   private:
    uint8_t cur_[MAX_PAYLOAD];
    uint8_t ready_[MAX_PAYLOAD];
    uint32_t cur_id_ = 0, done_id_ = 0, cur_mask_ = 0;
    uint16_t cur_len_  = 0;
    size_t ready_len_  = 0;
    portMUX_TYPE lock_ = portMUX_INITIALIZER_UNLOCKED;
};

}  // namespace face24link
