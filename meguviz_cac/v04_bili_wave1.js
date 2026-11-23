/**
 * In-browser Bilibili Wave collector for MeguvizCAC v0.4 (16 partitions x 4 eras).
 * Runs inside a user-logged-in https://www.bilibili.com/ page via __biliV04Start(opts).
 * Strictly strips all author nicknames, UIDs, and IP locations; stops cleanly on any -352/-412 gate.
 */
(function (root) {
  'use strict';

  const MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
  ];

  const CONTACT_RE = /[@＠]|https?:\/\/|\bwww\.|\b1[3-9]\d{9}\b|手机号|身份证|家庭住址|加我微信|加QQ群/i;
  const HARD_GATE_CODES = new Set([-352, -412, -101, -403]);

  const PARTITIONS = [
    { name: '动画', rid: 1, kw: ['动画解析', '手书', '独立动画'] },
    { name: '番剧国创', rid: 167, kw: ['新番吐槽', '国创动画', '经典老番'] },
    { name: '游戏', rid: 4, kw: ['单机攻略', '独立游戏测评', '速通'] },
    { name: '知识', rid: 36, kw: ['科普公开课', '历史长视频', '物理讲解'] },
    { name: '科技数码', rid: 188, kw: ['显卡实测', '开源AI部署', 'NAS教程'] },
    { name: '生活', rid: 160, kw: ['独居Vlog', '手工DIY', '日常整活'] },
    { name: '美食', rid: 211, kw: ['家常菜教程', '美食探店', '烘焙实操'] },
    { name: '影视', rid: 181, kw: ['电影拉片', '悬疑剧拆解', '老剧重温'] },
    { name: '音乐', rid: 3, kw: ['编曲教学', '现场Live', '吉他指弹'] },
    { name: '鬼畜', rid: 119, kw: ['鬼畜调教', '人力Vocaloid', '音MAD'] },
    { name: '娱乐', rid: 5, kw: ['综艺名场面', '采访访谈', '脱口秀'] },
    { name: '资讯', rid: 202, kw: ['行业观察', '财报解读', '科技动态'] },
    { name: '汽车', rid: 223, kw: ['新能源续航实测', '二手车整备', '底盘对比'] },
    { name: '运动', rid: 234, kw: ['骑行记录', '羽毛球教学', '马拉松装备'] },
    { name: '动物', rid: 217, kw: ['流浪猫救助', '野保观察', '水族造景'] },
    { name: '时尚舞蹈', rid: 155, kw: ['街舞编舞', '国风穿搭', '宅舞'] },
  ];

  // RFC 1321 MD5 over UTF-8 bytes (self-contained, zero external deps).
  const MD5_S = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
  ];
  const MD5_K = Array.from({ length: 64 }, (_, i) => Math.floor(Math.abs(Math.sin(i + 1)) * 4294967296) >>> 0);

  function md5(str) {
    const raw = new TextEncoder().encode(String(str));
    const bitLen = raw.length * 8;
    const paddedLen = (((raw.length + 8) >>> 6) + 1) << 6;
    const buf = new Uint8Array(paddedLen);
    buf.set(raw);
    buf[raw.length] = 0x80;
    const dv = new DataView(buf.buffer);
    dv.setUint32(paddedLen - 8, bitLen >>> 0, true);
    dv.setUint32(paddedLen - 4, Math.floor(bitLen / 4294967296) >>> 0, true);

    let a0 = 0x67452301;
    let b0 = 0xefcdab89;
    let c0 = 0x98badcfe;
    let d0 = 0x10325476;

    for (let off = 0; off < paddedLen; off += 64) {
      let A = a0, B = b0, C = c0, D = d0;
      for (let i = 0; i < 64; i++) {
        let F, g;
        if (i < 16) {
          F = (B & C) | ((~B) & D);
          g = i;
        } else if (i < 32) {
          F = (D & B) | ((~D) & C);
          g = (5 * i + 1) & 15;
        } else if (i < 48) {
          F = B ^ C ^ D;
          g = (3 * i + 5) & 15;
        } else {
          F = C ^ (B | (~D));
          g = (7 * i) & 15;
        }
        const word = dv.getUint32(off + g * 4, true);
        const sum = (A + F + MD5_K[i] + word) >>> 0;
        const rot = ((sum << MD5_S[i]) | (sum >>> (32 - MD5_S[i]))) >>> 0;
        A = D;
        D = C;
        C = B;
        B = (B + rot) >>> 0;
      }
      a0 = (a0 + A) >>> 0;
      b0 = (b0 + B) >>> 0;
      c0 = (c0 + C) >>> 0;
      d0 = (d0 + D) >>> 0;
    }

    const out = [a0, b0, c0, d0];
    let hex = '';
    for (const w of out) {
      for (let b = 0; b < 4; b++) {
        hex += ((w >>> (b * 8)) & 0xff).toString(16).padStart(2, '0');
      }
    }
    return hex;
  }

  function encWbi(params, imgKey, subKey) {
    const orig = imgKey + subKey;
    const mixinKey = MIXIN_KEY_ENC_TAB.map((n) => orig[n] || '').join('').slice(0, 32);
    const full = { ...params, wts: Math.floor(Date.now() / 1000) };
    const query = Object.keys(full)
      .sort()
      .map((k) => {
        const val = String(full[k]).replace(/[!'()*]/g, '');
        return `${encodeURIComponent(k)}=${encodeURIComponent(val)}`;
      })
      .join('&');
    const w_rid = md5(query + mixinKey);
    return `${query}&w_rid=${w_rid}`;
  }

  function eraFromTs(ts) {
    const yr = new Date((Number(ts) || 0) * 1000).getUTCFullYear();
    if (yr > 0 && yr <= 2022) return 'pre_llm';
    if (yr >= 2023 && yr <= 2024) return 'early_llm';
    if (yr === 2025) return 'mid_llm';
    return 'current_2026';
  }

  function extractKeyFromUrl(url) {
    if (!url) return '';
    const m = String(url).match(/\/([^\/?#]+)\.\w+(?:[?#]|$)/);
    return m ? m[1] : '';
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function cleanText(raw) {
    if (typeof raw !== 'string') return '';
    const txt = raw.replace(/\s+/g, ' ').trim();
    if (!txt || CONTACT_RE.test(txt)) return '';
    return txt;
  }

  function triggerDownload(filename, lines) {
    const blob = new Blob([lines.join('\n') + '\n'], { type: 'application/x-ndjson;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2500);
  }

  async function runBiliV04Wave(options = {}) {
    const wave = options.wave || 'w1';
    const targetRows = Number(options.targetRows ?? 40000);
    const maxMinutes = Number(options.maxMinutes ?? 14);
    const flushEvery = Number(options.flushEvery ?? 5000);
    const maxPerVideo = Number(options.maxPerVideo ?? 450);
    const captureDate = new Date().toISOString().slice(0, 10);
    const deadline = Date.now() + maxMinutes * 60_000;

    const state = {
      wave,
      startedAt: new Date().toISOString(),
      running: true,
      done: false,
      status: 'running',
      totalRows: 0,
      chunksDownloaded: 0,
      downloadedFiles: [],
      videosVisited: 0,
      byPartition: {},
      byEra: {},
      lastBvid: null,
      stopCode: null,
      searchSkipped: 0,
      error: null,
    };
    if (root && typeof root === 'object') {
      root.__biliV04State = state;
    }

    const buffer = [];
    const seenComments = new Set();
    const seenVideos = new Set();

    const flush = (force = false) => {
      if (!buffer.length || (!force && buffer.length < flushEvery)) return;
      state.chunksDownloaded += 1;
      const fname = `bili_v04_${wave}_chunk_${String(state.chunksDownloaded).padStart(3, '0')}.jsonl`;
      const chunk = buffer.splice(0, buffer.length);
      triggerDownload(fname, chunk);
      state.downloadedFiles.push({ file: fname, rows: chunk.length });
    };

    const apiGet = async (url) => {
      await sleep(350 + Math.floor(Math.random() * 200));
      const resp = await fetch(url, { credentials: 'include' });
      if (resp.status === 412) {
        return { gate: true, code: 412, json: null };
      }
      if (!resp.ok) {
        return { gate: false, code: resp.status, json: null };
      }
      const j = await resp.json();
      if (j && HARD_GATE_CODES.has(j.code)) {
        return { gate: true, code: j.code, json: j };
      }
      return { gate: false, code: j?.code ?? 0, json: j };
    };

    try {
      const nav = await apiGet('https://api.bilibili.com/x/web-interface/nav');
      if (nav.gate) {
        state.status = 'stopped_on_gate';
        state.stopCode = nav.code;
        return finish();
      }
      const navData = nav.json?.data;
      if (!navData?.isLogin) {
        state.status = 'login_required';
        return finish();
      }
      const imgKey = extractKeyFromUrl(navData.wbi_img?.img_url);
      const subKey = extractKeyFromUrl(navData.wbi_img?.sub_url);

      const eraWindows = [
        { name: 'pre_llm', begin: 1546272000, end: 1672502399 },
        { name: 'early_llm', begin: 1672502400, end: 1735660799 },
        { name: 'mid_llm', begin: 1735660800, end: 1767196799 },
        { name: 'current_2026', begin: 1767196800, end: Math.floor(Date.now() / 1000) },
      ];

      const addRow = (video, rawReply, level, parentId, sortLabel) => {
        if (!rawReply || !rawReply.rpid) return false;
        const rpid = String(rawReply.rpid);
        if (seenComments.has(rpid)) return false;
        const txt = cleanText(rawReply.content?.message);
        if (!txt) return false;
        seenComments.add(rpid);
        const ts = Number(rawReply.ctime || 0) || null;
        const era = eraFromTs(ts || video.pubdate || 0);
        const rec = {
          bvid: video.bvid,
          partition: video.partition,
          video_title: video.title,
          pubdate: video.pubdate,
          comment_level: level,
          parent_id: parentId || null,
          text: txt,
          posted_ts: ts,
          like_count: typeof rawReply.like === 'number' ? rawReply.like : null,
          is_pinned: Boolean(rawReply.card_label?.some?.((c) => /置顶/.test(c?.text_content || ''))),
          sort: sortLabel,
          capture_date: captureDate,
        };
        buffer.push(JSON.stringify(rec));
        state.totalRows += 1;
        state.byPartition[video.partition] = (state.byPartition[video.partition] || 0) + 1;
        state.byEra[era] = (state.byEra[era] || 0) + 1;
        if (buffer.length >= flushEvery) flush(true);
        return true;
      };

      const fetchVideoComments = async (video) => {
        if (Date.now() >= deadline || state.totalRows >= targetRows) return 'done';
        let aid = video.aid;
        let pubdate = video.pubdate || null;
        let title = video.title || '';
        if (!aid) {
          const vw = await apiGet(`https://api.bilibili.com/x/web-interface/view?bvid=${encodeURIComponent(video.bvid)}`);
          if (vw.gate) {
            state.stopCode = vw.code;
            return 'gate';
          }
          aid = vw.json?.data?.aid;
          pubdate = pubdate || vw.json?.data?.pubdate || null;
          title = title || (vw.json?.data?.title || '');
        }
        if (!aid) return 'skip';
        const vmeta = { bvid: video.bvid, aid, partition: video.partition, title: String(title).replace(/\s+/g, ' ').trim().slice(0, 260), pubdate };
        state.videosVisited += 1;
        state.lastBvid = vmeta.bvid;

        let videoCount = 0;
        let offset = '';
        let useClassic = false;
        let pn = 1;
        let pageGuard = 0;

        while (pageGuard < 22 && videoCount < maxPerVideo && Date.now() < deadline && state.totalRows < targetRows) {
          pageGuard += 1;
          let res;
          if (!useClassic && imgKey && subKey) {
            const qs = encWbi({
              oid: aid,
              type: 1,
              mode: 3,
              plat: 1,
              web_location: '1315875',
              pagination_str: JSON.stringify({ offset }),
            }, imgKey, subKey);
            res = await apiGet(`https://api.bilibili.com/x/v2/reply/wbi/main?${qs}`);
            if (res.gate) {
              state.stopCode = res.code;
              return 'gate';
            }
            if (res.code !== 0 || !res.json?.data) {
              useClassic = true;
              continue;
            }
          } else {
            res = await apiGet(`https://api.bilibili.com/x/v2/reply?type=1&oid=${aid}&pn=${pn}&ps=20&sort=2`);
            if (res.gate) {
              state.stopCode = res.code;
              return 'gate';
            }
            if (res.code !== 0 || !res.json?.data) break;
            pn += 1;
          }

          const d = res.json.data;
          const topList = [];
          if (pageGuard === 1) {
            if (Array.isArray(d.top_replies)) topList.push(...d.top_replies);
            if (d.top?.upper) topList.push(d.top.upper);
          }
          const mainList = Array.isArray(d.replies) ? d.replies : [];
          const combined = [...topList, ...mainList];
          if (!combined.length) break;

          const needReplyThreads = [];
          for (const item of combined) {
            if (videoCount >= maxPerVideo || state.totalRows >= targetRows) break;
            if (addRow(vmeta, item, 'top', null, useClassic ? 'classic_hot' : 'wbi_main')) {
              videoCount += 1;
            }
            if (Array.isArray(item.replies)) {
              for (const sub of item.replies) {
                if (videoCount >= maxPerVideo || state.totalRows >= targetRows) break;
                if (addRow(vmeta, sub, 'reply', String(item.rpid), useClassic ? 'classic_embedded' : 'wbi_embedded')) {
                  videoCount += 1;
                }
              }
            }
            if ((item.rcount || 0) > 3 && needReplyThreads.length < 4) {
              needReplyThreads.push(item.rpid);
            }
          }

          for (const rootRpid of needReplyThreads) {
            if (videoCount >= maxPerVideo || Date.now() >= deadline || state.totalRows >= targetRows) break;
            const subRes = await apiGet(`https://api.bilibili.com/x/v2/reply/reply?type=1&oid=${aid}&root=${rootRpid}&pn=1&ps=20`);
            if (subRes.gate) {
              state.stopCode = subRes.code;
              return 'gate';
            }
            for (const sub of (subRes.json?.data?.replies || [])) {
              if (videoCount >= maxPerVideo || state.totalRows >= targetRows) break;
              if (addRow(vmeta, sub, 'reply', String(rootRpid), 'reply_p1')) {
                videoCount += 1;
              }
            }
          }

          if (!useClassic) {
            const nextOff = d.cursor?.pagination_reply?.next_offset;
            if (!nextOff || d.cursor?.is_end) break;
            offset = nextOff;
          } else {
            if (mainList.length < 20) break;
          }
        }
        return 'ok';
      };

      const queues = new Map(PARTITIONS.map((p) => [p.name, []]));

      // Round-robin discovery + pull across all 16 partitions and 4 historical eras.
      for (let round = 0; round < 6 && Date.now() < deadline && state.totalRows < targetRows; round++) {
        for (let pIdx = 0; pIdx < PARTITIONS.length && Date.now() < deadline && state.totalRows < targetRows; pIdx++) {
          const part = PARTITIONS[pIdx];
          const q = queues.get(part.name);

          if (q.length < 3) {
            if (round === 0) {
              const rk = await apiGet(`https://api.bilibili.com/x/web-interface/ranking/v2?rid=${part.rid}&type=all`);
              if (rk.gate) { state.stopCode = rk.code; return finish('stopped_on_gate'); }
              for (const v of (rk.json?.data?.list || [])) {
                if (v?.bvid && !seenVideos.has(v.bvid)) {
                  seenVideos.add(v.bvid);
                  q.push({ bvid: v.bvid, aid: v.aid, title: v.title, pubdate: v.pubdate, partition: part.name });
                }
              }
            }

            const dyn = await apiGet(`https://api.bilibili.com/x/web-interface/dynamic/region?rid=${part.rid}&ps=25&pn=${round + 1}`);
            if (dyn.gate) { state.stopCode = dyn.code; return finish('stopped_on_gate'); }
            for (const v of (dyn.json?.data?.archives || [])) {
              if (v?.bvid && !seenVideos.has(v.bvid)) {
                seenVideos.add(v.bvid);
                q.push({ bvid: v.bvid, aid: v.aid, title: v.title, pubdate: v.pubdate, partition: part.name });
              }
            }

            if (imgKey && subKey) {
              const eraWin = eraWindows[(round + pIdx) % eraWindows.length];
              const keyword = part.kw[(round + pIdx) % part.kw.length];
              const qs = encWbi({
                search_type: 'video',
                keyword,
                order: round % 2 === 0 ? 'click' : 'dm',
                page: Math.floor(round / 2) + 1,
                pubtime_begin_s: eraWin.begin,
                pubtime_end_s: eraWin.end,
              }, imgKey, subKey);
              const sr = await apiGet(`https://api.bilibili.com/x/web-interface/wbi/search/type?${qs}`);
              if (sr.gate) { state.stopCode = sr.code; return finish('stopped_on_gate'); }
              if (sr.code === 0 && Array.isArray(sr.json?.data?.result)) {
                for (const v of sr.json.data.result) {
                  if (v?.bvid && !seenVideos.has(v.bvid)) {
                    seenVideos.add(v.bvid);
                    q.push({
                      bvid: v.bvid,
                      aid: v.aid || v.id,
                      title: String(v.title || '').replace(/<[^>]+>/g, ''),
                      pubdate: v.pubdate,
                      partition: part.name,
                    });
                  }
                }
              } else {
                state.searchSkipped += 1;
              }
            }
          }

          // Visit 2 videos per partition per round to stay well-balanced across all 16 partitions.
          for (let take = 0; take < 2 && q.length && Date.now() < deadline && state.totalRows < targetRows; take++) {
            const vid = q.shift();
            const outcome = await fetchVideoComments(vid);
            if (outcome === 'gate') {
              return finish('stopped_on_gate');
            }
          }
        }
      }

      return finish(state.totalRows >= targetRows ? 'target_reached' : Date.now() >= deadline ? 'time_budget_reached' : 'exhausted');
    } catch (err) {
      state.error = String(err?.message || err).slice(0, 220);
      return finish('error');
    }

    function finish(finalStatus) {
      if (finalStatus) state.status = finalStatus;
      flush(true);
      state.running = false;
      state.done = true;
      state.finishedAt = new Date().toISOString();
      return state;
    }
  }

  if (root && typeof root === 'object') {
    root.runBiliV04Wave = runBiliV04Wave;
    root.__biliV04Start = function (opts) {
      if (root.__biliV04State && root.__biliV04State.running) {
        return { started: false, reason: 'already_running', state: root.__biliV04State };
      }
      root.__biliV04Promise = runBiliV04Wave(opts || {});
      return { started: true };
    };
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { md5, encWbi, runBiliV04Wave, PARTITIONS };
  }
})(typeof window !== 'undefined' ? window : globalThis);
