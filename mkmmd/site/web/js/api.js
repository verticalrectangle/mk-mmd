// The server's API: every path is relative to the page (`/<token>/`), so the token travels with it.

export async function api(method, name, { query = {}, body } = {}) {
  const qs = new URLSearchParams(query).toString();
  const init = { method };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  const res = await fetch(`api/${name}${qs ? "?" + qs : ""}`, init);
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

export const fileUrl = (path) => "file?path=" + encodeURIComponent(path);

export async function bytes(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let why = `${res.status} ${res.statusText}`;
    try { why = (await res.json()).error || why; } catch (e) { /* not JSON */ }
    throw new Error(why);
  }
  return res.arrayBuffer();
}
