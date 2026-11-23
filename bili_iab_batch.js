/**
 * Batch-read visible comments through a CUA in-app browser tab.
 * The journal callbacks receive only the observed comments and progress.
 */
export function createBiliIabBatch(video, journal, captureDate) {
    const threads = video.playwright.locator('#commentapp').locator('bili-comment-thread-renderer');
    const body = video.playwright.locator('body');
    const continuation = video.playwright.locator('#commentapp').locator('bili-comments').locator('#continuations');
    const replyProgress = new Map();
    let current;

    function capture({from = 0, index = null} = {}) {
        return video.playwright.evaluate(({from, index}) => {
            const section = document.getElementById('commentapp');
            const root = section?.querySelector('bili-comments')?.shadowRoot;
            const visible = el => !!(el && el.getBoundingClientRect().height > 0 && getComputedStyle(el).visibility !== 'hidden');
            const gate = root?.getElementById('limit-mask');
            const tail = root?.getElementById('continuations');
            const end = root?.getElementById('end');
            const feed = root?.getElementById('contents');
            const notice = [document.body?.innerText?.slice(0, 340), visible(gate) ? gate.innerText : '', visible(tail) ? tail.innerText : ''].join(' ');
            const blocked = /安全风控|错误号.{0,3}412|请求过于频繁|访问过于频繁|系统检测到异常|请进行安全验证/.test(notice);
            const login = (visible(gate) && /登录后查看/.test(gate.innerText)) ||
                [...document.querySelectorAll('input[type=password]')].some(visible);
            const elements = [...(feed?.querySelectorAll('bili-comment-thread-renderer') ?? [])];

            function flatten(node) {
                if (node.nodeType === 3) return node.textContent ?? '';
                if (node.nodeType !== 1) return '';
                if (node.localName === 'br') return ' ';
                if (node.localName === 'img') {
                    const alt = node.getAttribute('alt') ?? '';
                    return /^\[[^\[\]\r\n]{1,22}\]$/.test(alt) ? alt : '';
                }
                if (node.localName === 'a' && (/^[@＠]/.test(node.innerText.trim()) ||
                    node.getAttribute('href')?.includes('space.bilibili.com'))) return '';
                return [...node.childNodes].map(flatten).join('');
            }

            function readComment(renderer, reply) {
                const tree = renderer?.shadowRoot;
                const body = tree?.getElementById(reply ? 'main' : 'content');
                const content = body?.querySelector('bili-rich-text')?.shadowRoot?.getElementById('contents');
                if (!visible(content)) return null;
                const text = flatten(content).replace(/\s+/g, ' ').trim();
                const time = tree?.getElementById('footer')?.querySelector('bili-comment-action-buttons-renderer')
                    ?.shadowRoot?.getElementById('pubdate')?.innerText?.split(/IP\s*[属屬]地/)[0].trim();
                if (!text || !time || /[@＠]|https?:\/\/|\bwww\.|\b1[3-9]\d{9}\b|手机号|身份证|家庭住址|加我微信|加QQ群/i.test(text)) return null;
                const header = reply ? body : tree?.getElementById('header');
                const name = header?.querySelector('bili-comment-user-info')?.shadowRoot?.getElementById('user-name')?.querySelector('a');
                const nickname = visible(name) ? name.innerText.trim() : '';
                const fatFish = /大肥[鱼魚]/.test(nickname);
                const whiteRice = /白[饭飯]/.test(nickname);
                const legacyFish = /大胖[鱼魚]/.test(nickname);
                const hint = fatFish || legacyFish ? '大肥鱼类' : /[鱼魚]/.test(nickname) ? '含鱼' : whiteRice ? '含白饭' : null;
                const rule = fatFish && whiteRice ? '大肥鱼和白饭' : fatFish ? '大肥鱼' : whiteRice ? '白饭' : null;
                return {
                    text, displayed_time: time, checked_name: !!nickname, name_hint: hint, name_rule: rule,
                    pinned: !reply && visible(body?.querySelector('#top')),
                };
            }

            function readThread(thread, position) {
                const top = readComment(thread.shadowRoot?.getElementById('comment'), false);
                const replies = thread.shadowRoot?.querySelector('bili-comment-replies-renderer')?.shadowRoot;
                const replyHosts = [...(replies?.getElementById('expander-contents')?.querySelectorAll('bili-comment-reply-renderer') ?? [])].filter(visible);
                const pager = replies?.getElementById('pagination');
                const more = replies?.getElementById('view-more');
                const buttons = [...(pager?.querySelector('#pagination-body')?.children ?? [])];
                const pages = Number(pager?.querySelector('#pagination-head')?.innerText?.match(/共\s*(\d+)\s*页/)?.[1] ?? 0);
                const pageButton = buttons.find(el => el.getAttribute('style')?.includes('--brand_blue'));
                const page = Number(pageButton?.innerText ?? (pages === 1 ? 1 : 0));
                return {
                    position, top, rawReplies: replyHosts.length, page, pages, paged: visible(pager),
                    more: visible(more) && /点击查看/.test(more.innerText),
                    advertisedReplies: Number(more?.innerText?.match(/共\s*(\d+)\s*条回复/)?.[1] ?? 0),
                    next: buttons.some(el => el.innerText.trim() === '下一页'),
                    numericPages: buttons.map(el => Number(el.innerText.trim())).filter(Number.isInteger),
                    replies: top ? replyHosts.map(node => readComment(node, true)).filter(Boolean)
                        .map(row => ({...row, comment_level: 'reply', parent_text: top.text})) : [],
                };
            }

            let entries = [];
            if (!blocked && !login) {
                entries = index === null ? elements.slice(from).map((thread, offset) => readThread(thread, from + offset)) :
                    elements[index] ? [readThread(elements[index], index)] : [];
            }
            return {
                bvid: location.pathname.split('/').find(part => /^BV[A-Za-z0-9]{10}$/.test(part)),
                blocked, login, height: innerHeight, width: innerWidth, rendered: elements.length, entries,
                commentTop: section?.getBoundingClientRect().top ?? null,
                pagerReady: visible(tail) && tail.getBoundingClientRect().height >= 12 &&
                    !tail.querySelector('form,textarea,a,button,input,[role=button]'),
                title: (document.querySelector('#viewbox_report h1') ?? document.querySelector('h1'))?.innerText?.trim(),
                ended: (visible(end) && /没有更多/.test(end.innerText)) ||
                    (visible(tail) && /没有更多评论了/.test(tail.innerText)),
                empty: elements.length === 0 && /还没有评论|暂无评论/.test(feed?.innerText ?? ''),
            };
        }, {from, index}, {timeout: 4200});
    }

    async function waitForThread(position, timeout) {
        try { await threads.nth(position).waitFor({state: 'attached', timeout}); }
        catch { /* The next DOM read distinguishes an empty or slow list from a restriction. */ }
    }

    function addThread(state, thread) {
        if (!thread?.top) {
            if (thread?.more || thread?.rawReplies) state.inaccessible.add(thread.position);
            return;
        }
        const key = row => JSON.stringify([row.parent_text ?? null, row.text, row.name_rule]);
        for (const row of [thread.top, ...thread.replies]) {
            const id = key(row);
            if (!state.seen.has(id)) {
                state.seen.add(id);
                state.rows.push(row);
            }
        }
        if (thread.more || thread.paged) state.waiting.set(thread.top.text, thread.position);
    }

    async function save(state, repliesEnded = false) {
        if (!state.rows.length && !state.pages.size && state.ended === state.savedEnd && repliesEnded === state.savedReplyEnd) return;
        const batch = {
            bvid: state.source.bvid, url: state.source.url, title: state.title, sort: 'page_default',
            capture_date: captureDate, sampled_from_channel: state.source.sampled_from_channel,
            channel_source_url: state.source.channel_source_url, rows: state.rows, reply_pages: [...state.pages.values()],
            ended: state.ended, replies_ended: repliesEnded,
        };
        await journal.append(batch);
        state.captured += batch.rows.length;
        state.rows = [];
        state.pages.clear();
        state.savedEnd = batch.ended;
        state.savedReplyEnd = repliesEnded;
        state.savedAt = Date.now();
    }

    function markReplyPage(state, parent, page, complete) {
        state.progress.set(parent, page);
        state.pages.set(parent, {parent_text: parent, page, complete});
        if (complete) state.finished.add(parent);
    }

    async function expandReplies(state, parent, index, deadline) {
        const thread = threads.nth(index);
        const read = () => capture({index});
        let view = await read();
        let entry = view.entries[0];
        if (view.blocked || view.login || entry?.top?.text !== parent) return view;
        const advertised = entry.advertisedReplies;
        if (entry.more && !entry.paged) {
            await thread.locator('#view-more').getByText('点击查看', {exact: true}).click({timeout: 4200});
            try { await thread.locator('#view-more').waitFor({state: 'hidden', timeout: 1500}); }
            catch { /* The next observation records whether rendering finished. */ }
            view = await read();
            entry = view.entries[0];
        }

        if (!view.blocked && !view.login && entry?.top?.text === parent && !entry.more &&
            !entry.paged && entry.rawReplies >= Math.max(advertised, 1)) {
            addThread(state, entry);
            markReplyPage(state, parent, 1, true);
            return view;
        }

        let stored = state.progress.get(parent) ?? 0;
        const clickPage = async label => {
            await thread.locator('#pagination-body').getByText(label, {exact: true}).click({timeout: 4200});
            await waitForThread(index, 1200);
            for (let retries = 0; retries < 3; retries++) {
                view = await read();
                const next = view.entries[0];
                if (view.blocked || view.login || (next?.page !== entry?.page && next?.rawReplies)) {
                    entry = next;
                    return;
                }
            }
            entry = view.entries[0];
        };
        if (entry?.page < stored && entry.numericPages.includes(stored + 1)) await clickPage(String(stored + 1));
        while (!view.blocked && !view.login && entry?.top?.text === parent && entry.paged && entry.page && entry.rawReplies) {
            addThread(state, entry);
            if (entry.page === stored + 1) {
                stored = entry.page;
                markReplyPage(state, parent, stored, entry.pages > 0 && stored === entry.pages);
            }
            if (state.finished.has(parent) || !entry.next || Date.now() >= deadline) break;
            const currentPage = entry.page;
            await clickPage('下一页');
            if (entry?.page === currentPage) break;
            if (state.rows.length >= 120 || Date.now() - state.savedAt > 8500) await save(state);
        }
        return view;
    }

    function matchesVideo(address, bvid) {
        if (!address) return false;
        const url = new URL(address);
        return url.origin === 'https://www.bilibili.com' &&
            new RegExp('^/(?:zh-Hans/)?video/' + bvid + '/?$').test(url.pathname);
    }

    async function visit(source, deadline, maxReplies, prioritizeReplies) {
        if (!/^BV[A-Za-z0-9]{10}$/.test(source.bvid) || !matchesVideo(source.url, source.bvid)) throw Error('未核验的视频链接');
        const active = matchesVideo(await video.url(), source.bvid);
        if (current && current.source.bvid !== source.bvid) await save(current, current.savedReplyEnd);
        if (!active) await video.goto(source.url);
        let view = await capture({index: -1});
        if (view.bvid === source.bvid && !view.blocked && !view.login && !view.rendered && !view.empty) {
            const heading = video.playwright.locator('#commentapp').getByText('评论', {exact: true});
            try {
                await video.getAXState({emit: false});
                await heading.waitFor({state: 'visible', timeout: 2600});
                if (await heading.count() === 1) {
                    await heading.click({timeout: 3300});
                    await waitForThread(0, 2150);
                }
            } catch { /* The subsequent viewport check may still recover the page. */ }
            view = await capture({index: -1});
        }
        for (let step = 0; step < 4 && view.bvid === source.bvid && !view.blocked && !view.login && !view.rendered && !view.empty; step++) {
            const top = view.commentTop;
            if (top === null || top > view.height - 100) await body.press('PageDown', {timeout: 3200});
            else if (top < -95) await body.press('PageUp', {timeout: 3200});
            await waitForThread(0, top !== null && top >= -95 && top <= view.height - 100 ? 1300 : 750);
            view = await capture({index: -1});
        }
        if (view.bvid === source.bvid && !view.blocked && !view.login && !view.rendered &&
            !view.empty && view.commentTop !== null) {
            const target = new URL(await video.url());
            if (matchesVideo(target.href, source.bvid) && target.hash !== '#commentapp') {
                target.hash = 'commentapp';
                await video.goto(target.href);
                await waitForThread(0, 2300);
                view = await capture({index: -1});
            }
        }
        if (view.blocked || view.login || view.bvid !== source.bvid || (!view.rendered && !view.empty)) {
            return {bvid: source.bvid, blocked: view.blocked, login: view.login, rendered: view.rendered,
                    status: view.blocked ? 'access_gate' : view.login ? 'login_required' :
                        view.bvid !== source.bvid ? 'navigation_incomplete' : 'comments_not_rendered'};
        }
        if (!active || current?.source.bvid !== source.bvid) {
            if (!replyProgress.has(source.bvid)) {
                const entries = journal.replyPages?.[source.bvid] ?? [];
                replyProgress.set(source.bvid, {
                    progress: new Map(entries.map(([parent, page]) => [parent, page])),
                    finished: new Set(entries.filter(([, , complete]) => complete).map(([parent]) => parent)),
                });
            }
            current = {
                source, title: view.title, rows: [], seen: new Set(), pages: new Map(), waiting: new Map(),
                ...replyProgress.get(source.bvid),
                inaccessible: new Set(), savedAt: Date.now(), captured: 0, offset: 0,
                ended: false, savedEnd: false, savedReplyEnd: false,
            };
        }
        const state = current;
        const before = state.captured;
        const attempts = new Set();
        let stalled = 0, replyError = false, useWheel = false, triedContinuation = false;
        view = await capture({from: state.offset});
        while (view.bvid === source.bvid && !view.blocked && !view.login) {
            for (const entry of view.entries) addThread(state, entry);
            state.offset = Math.max(state.offset, view.rendered);
            state.ended ||= view.ended || view.empty;
            if (state.rows.length >= 120 || Date.now() - state.savedAt > 8500) await save(state);

            const waiting = [...state.waiting].find(([parent]) => !state.finished.has(parent) && !attempts.has(parent));
            if (waiting && attempts.size < maxReplies && Date.now() + 750 < deadline &&
                (state.ended || prioritizeReplies || (state.offset >= 38 && attempts.size === 0))) {
                attempts.add(waiting[0]);
                try {
                    const replies = await expandReplies(state, ...waiting, Math.min(deadline, Date.now() + 5200));
                    if (replies.blocked || replies.login) {
                        view = replies;
                        break;
                    }
                } catch {
                    replyError = true;
                    const activeView = await capture({index: -1});
                    if (activeView.blocked || activeView.login || activeView.bvid !== source.bvid) {
                        view = activeView;
                        break;
                    }
                }
                if (state.ended) continue;
            }
            if (state.ended || Date.now() >= deadline || stalled >= 2) break;
            const count = view.rendered;
            if (!useWheel && stalled === 0) {
                try {
                    await body.press('End', {timeout: 3600});
                } catch (error) {
                    if (Date.now() + 1800 >= deadline) throw error;
                    const afterKey = await capture({from: state.offset});
                    if (afterKey.blocked || afterKey.login) {
                        view = afterKey;
                        break;
                    }
                    if (afterKey.bvid !== source.bvid) throw error;
                    useWheel = true;
                    if (afterKey.rendered > count || afterKey.ended) {
                        view = afterKey;
                        continue;
                    }
                }
            }
            if (useWheel || stalled > 0) {
                let positioned = false;
                if (!triedContinuation && view.pagerReady && Date.now() + 2100 < deadline) {
                    triedContinuation = true;
                    try {
                        await continuation.click({timeout: 2400});
                        positioned = true;
                    } catch {
                        view = await capture({from: state.offset});
                        if (view.blocked || view.login || view.bvid !== source.bvid) break;
                        if (view.rendered > count || view.ended) {
                            stalled = 0;
                            continue;
                        }
                    }
                }
                if (!positioned) {
                    await video.scroll([Math.floor(view.width * 0.40), Math.floor(view.height * 0.80)], 'down', 7);
                }
            }
            await waitForThread(count, 1250);
            view = await capture({from: state.offset});
            stalled = view.rendered <= count && !view.ended ? stalled + 1 : 0;
        }
        const replyEnd = state.ended && !view.blocked && !view.login && state.inaccessible.size === 0 &&
            [...state.waiting.keys()].every(parent => state.finished.has(parent));
        await save(state, replyEnd);
        return {
            bvid: source.bvid, captured: state.captured - before, rendered: view.rendered,
            topEnd: state.ended, replyEnd, blocked: view.blocked, login: view.login,
            status: replyEnd ? 'observed_both_ends' : replyError ? 'partial_reply_timeout' :
                state.ended ? 'observed_top_end' : stalled >= 2 ? 'partial_list_stalled' : 'partial',
        };
    }

    return {
        async run(videos, {budgetMs = 42000, perVideoMs = 10000, maxReplies = 3, prioritizeReplies = false} = {}) {
            const deadline = Date.now() + budgetMs;
            const reports = [];
            for (const source of videos) {
                if (Date.now() + 3200 >= deadline) break;
                let report;
                try {
                    report = await visit(source, Math.min(deadline, Date.now() + perVideoMs), maxReplies, prioritizeReplies);
                } catch (error) {
                    if (current?.rows.length || current?.pages.size) await save(current, current.savedReplyEnd);
                    let gate;
                    try { gate = await capture({index: -1}); } catch { /* The timed-out tab may be busy. */ }
                    report = {bvid: source.bvid, status: 'interaction_timeout', blocked: gate?.blocked === true,
                              login: gate?.login === true, error: String(error).slice(0, 180)};
                }
                await journal.attempt({...report, capture_date: captureDate, queue_round: source.queue_round});
                reports.push(report);
                if (report.blocked || report.login) break;
            }
            return reports;
        },
    };
}
