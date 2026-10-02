// Client event handlers tested without a browser or cloud generation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function client() {
    const streams = [];
    const elements = new Map();
    function element() {
        const el = {
            children: [], textContent: '', classList: { add() {}, remove() {} },
            appendChild(row) { row.parent = this; this.children.push(row); },
            remove() { this.parent.children.shift(); },
        };
        Object.defineProperty(el, 'firstElementChild', { get: () => el.children[0] });
        return el;
    }
    class FakeEventSource {
        static CONNECTING = 0;
        static CLOSED = 2;
        constructor(url) { this.url = url; this.readyState = 0; this.closed = false; streams.push(this); }
        close() { this.closed = true; this.readyState = 2; }
    }
    const context = vm.createContext({
        EventSource: FakeEventSource, console, setTimeout,
        document: {
            addEventListener() {}, querySelectorAll: () => [],
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, element());
                return elements.get(id);
            },
            createElement: element,
        },
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/app.js'), 'utf8'), context);
    vm.runInContext(`
        currentProject = 'sse_test';
        showToast = () => {};
        loadProjects = () => {};
        loadCombineVersions = () => {};
        markAllPipelineNodesDone = () => {};
    `, context);
    return { context, streams, elements };
}

test('a temporary disconnect preserves generation and native reconnect', () => {
    const { context, streams } = client();
    vm.runInContext('isGenerating = true; connectSSEStream();', context);
    const stream = streams[0];
    stream.onerror();
    assert.equal(stream.closed, false);
    assert.equal(vm.runInContext('isGenerating', context), true);
    stream.onopen();
    stream.onmessage({ data: JSON.stringify({ type: 'done', payload: {} }) });
    assert.equal(stream.closed, true);
    assert.equal(vm.runInContext('isGenerating', context), false);
});

test('an old stream error cannot close a replacement stream', () => {
    const { context, streams } = client();
    vm.runInContext('connectSSEStream(); connectSSEStream();', context);
    streams[0].onerror();
    assert.equal(streams[1].closed, false);
    assert.equal(vm.runInContext('eventSource !== null', context), true);
});

test('combine transport errors keep native reconnect until completion', () => {
    const { context, streams } = client();
    vm.runInContext('connectCombineStream();', context);
    streams[0].onerror();
    assert.equal(streams[0].closed, false);
    streams[0].onmessage({ data: JSON.stringify({ type: 'combine_done', payload: {} }) });
    assert.equal(streams[0].closed, true);
});

test('token fragments do not flood the terminal and log history stays bounded', () => {
    const { context, elements } = client();
    vm.runInContext(`
        for (let i = 0; i < 10000; i++) appendLogEntry({type: 'token', payload: 'a'});
        for (let i = 0; i < 1000; i++) appendLogEntry({type: 'status', payload: String(i)});
    `, context);
    const terminal = elements.get('logTerminal');
    assert.equal(terminal.children.length, 500);
    assert.match(terminal.children[0].innerHTML, />500</);
    assert.match(terminal.children[499].innerHTML, />999</);
});
