# Cloudflare-in-front runbook (for future enablement)

> **Status:** Documented, not yet enabled. Current DNS is at **Bluehost**;
> the site is served directly by **GitHub Pages + Fastly**. This runbook
> is what we'd follow when we decide to put Cloudflare in front of the
> same GitHub Pages origin to unlock two agent-readiness checks:
>
> 1. `Link:` HTTP response headers (RFC 8288) — isitagentready.com
> 2. `Accept: text/markdown` content negotiation — buildwithfern.com,
>    isitagentready.com
>
> Neither requires changing hosting. Cloudflare only proxies the existing
> origin (GitHub Pages) and adds response-layer logic.

## What this unlocks

| Scanner check | Before | After |
| --- | --- | --- |
| isitagentready — Link headers (RFC 8288) | ❌ | ✅ via Transform Rule |
| isitagentready — Markdown for Agents | ❌ | ✅ via Worker |
| buildwithfern — content-negotiation | ❌ | ✅ via Worker |

`markdown-url-support` (Fern's *separate* check for `/page.md` resolving)
is **already satisfied on pure GitHub Pages** by the
`sphinx-markdown-builder` CI step — no Cloudflare needed for that.

## Prerequisites

- Admin access to the Bluehost account holding the `eegdash.org`
  registration (or at least DNS zone access).
- A free Cloudflare account.
- Ability to edit the `eegdash/EEGDash` GitHub Pages settings (to keep
  CNAME verification working through the migration).

## Step 1 — Add the zone to Cloudflare

1. Cloudflare dashboard → **Add a site** → `eegdash.org` → Free plan.
2. Cloudflare scans the existing Bluehost zone and imports records.
   **Verify** the imported set before moving on:
    - `A`/`AAAA` records pointing at the GitHub Pages IPs
      (`185.199.108.153`, `185.199.109.153`, `185.199.110.153`,
      `185.199.111.153` and the IPv6 equivalents), **orange-clouded
      (proxied)**.
    - `CNAME` for `www.eegdash.org` → `eegdash.github.io`, proxied.
    - MX / TXT records for any email service — **keep grey-clouded
      (DNS-only)** so mail routing is untouched.
3. Note the two Cloudflare nameservers Cloudflare issues (example:
   `kari.ns.cloudflare.com`, `rex.ns.cloudflare.com` — Cloudflare picks
   fresh pair names per zone).

## Step 2 — Switch nameservers at Bluehost

1. Log in to Bluehost → **Domains → eegdash.org → Nameservers**.
2. Select **Use custom nameservers** and paste the two Cloudflare
   nameservers from Step 1.
3. Save. Bluehost shows a propagation warning (usually 1–24 h, often
   minutes). Cloudflare emails when activation completes.
4. **Do not remove the zone from Bluehost** until Cloudflare confirms
   active status — Bluehost keeps serving authoritative DNS until the
   glue records flip.

## Step 3 — Enforce HTTPS end-to-end

In Cloudflare → **SSL/TLS**:

- Encryption mode: **Full**. GitHub Pages serves valid TLS, so Full
  works without any certificate installation on the origin.
- **Edge Certificates → Always Use HTTPS**: on.
- **Automatic HTTPS Rewrites**: on.
- **HSTS**: start with 6 months, include subdomains, no preload.
  Preload only after running at max-age for a full cycle.

## Step 4 — Publish the `Link` HTTP header (RFC 8288)

Cloudflare → **Rules → Transform Rules → Modify Response Header → Create
rule**:

- **Rule name:** `Agent discovery Link header`
- **If incoming requests match:** custom filter
  `(http.request.uri.path eq "/" or ends_with(http.request.uri.path, "/index.html"))`
  — start with homepage only; expand to `and http.response.code == 200`
  once verified.
- **Then set:** response header `Link` with value:

```
</llms.txt>; rel="alternate"; type="text/markdown",
</.well-known/agent-skills/index.json>; rel="alternate"; type="application/json",
</.well-known/api-catalog>; rel="alternate"; type="application/linkset+json",
<https://data.eegdash.org/openapi.json>; rel="service-desc"; type="application/openapi+json"
```

(Commas separate entries; Cloudflare requires a single string — no
actual line breaks.)

After deploy:

```bash
curl -sI https://eegdash.org/ | grep -i '^link:'
```

should return the concatenated header.

## Step 5 — Content negotiation for `Accept: text/markdown`

Cloudflare → **Workers & Pages → Create Worker** named
`eegdash-md-negotiation`. Deploy the following script:

```js
// Route /<path> requests that ask for markdown to the /<path>.md sibling
// that we already publish via sphinx-markdown-builder. No-op for every
// other Accept header — GitHub Pages serves the HTML as before.
export default {
  async fetch(request) {
    const accept = request.headers.get("accept") || "";
    const wantsMarkdown =
      accept.includes("text/markdown") &&
      !accept.includes("text/html");

    const url = new URL(request.url);

    // Only rewrite real doc paths; leave assets, api, and the .well-known
    // directory to GitHub Pages untouched.
    const isDocPath =
      !url.pathname.startsWith("/_static/") &&
      !url.pathname.startsWith("/_images/") &&
      !url.pathname.startsWith("/.well-known/") &&
      !url.pathname.endsWith(".md") &&
      !url.pathname.endsWith(".txt") &&
      !url.pathname.endsWith(".xml") &&
      !url.pathname.endsWith(".json");

    if (wantsMarkdown && isDocPath) {
      let mdPath = url.pathname;
      if (mdPath.endsWith("/")) mdPath += "index.md";
      else if (mdPath.endsWith(".html")) mdPath = mdPath.replace(/\.html$/, ".md");
      else mdPath += ".md";

      url.pathname = mdPath;
      const mdResponse = await fetch(url.toString(), request);
      if (mdResponse.ok) {
        const headers = new Headers(mdResponse.headers);
        headers.set("content-type", "text/markdown; charset=utf-8");
        headers.set("vary", "Accept");
        return new Response(mdResponse.body, {
          status: mdResponse.status,
          headers,
        });
      }
    }

    return fetch(request);
  },
};
```

Bind the Worker to route `eegdash.org/*` (Workers → Routes → Add route).

Verify:

```bash
curl -sI -H 'Accept: text/markdown' https://eegdash.org/install/install.html \
  | grep -i '^content-type:'
# Expect: content-type: text/markdown; charset=utf-8
```

## Step 6 — Caching

- Cloudflare **Caching → Configuration**:
    - Browser Cache TTL: `Respect Existing Headers`
    - Crawler Hints: on
- **Page Rules / Cache Rules**: bypass cache on `/*.md` during the
  rollout week so we can iterate on the markdown output without
  flushing edge cache manually; tighten later.

## Step 7 — Verification checklist

Run after every Worker / Transform Rule change:

```bash
# Link header emitted
curl -sI https://eegdash.org/ | grep -i '^link:'

# Markdown negotiation works
diff \
  <(curl -s -H 'Accept: text/html' https://eegdash.org/install/install.html | head -c 200) \
  <(curl -s -H 'Accept: text/markdown' https://eegdash.org/install/install.html | head -c 200)
# Expect: different bodies

# HTML path still works for browsers
curl -sI https://eegdash.org/install/install.html \
  | grep -i '^content-type:'
# Expect: content-type: text/html; charset=utf-8

# Re-run the scanners and compare scores
open 'https://isitagentready.com/eegdash.org'
open 'https://buildwithfern.com/agent-score/company/eegdash'
```

## Step 8 — Rollback

Fast rollback (revert a single change):

- **Worker broken:** Cloudflare → Workers → Routes → delete the
  `eegdash.org/*` binding. Requests go straight to origin again.
- **Link header wrong:** Cloudflare → Transform Rules → disable the
  rule. No response-header modification.

Full rollback (drop Cloudflare entirely):

1. Re-enter Bluehost nameservers at the registrar.
2. Wait for NS propagation.
3. Remove the zone from Cloudflare.

GitHub Pages origin never changed, so rollback is always safe.

## Known caveats

- **Transform Rules character limit (1024 bytes).** The `Link` header
  value is well under that today, but adding more entries will
  eventually require splitting the rule or moving the header logic into
  the Worker.
- **MX-record email.** If `eegdash.org` serves any email address,
  verify the MX records stay DNS-only (grey cloud) — proxying them
  breaks SMTP.
- **GitHub Pages verification.** GitHub checks the zone's `CNAME` (or
  `A` records) to validate the custom domain. Proxying doesn't break
  this because Cloudflare still returns the correct values to GitHub's
  verifier over DNS; the orange cloud only affects the HTTP path.
- **AAAA addresses.** Keep both IPv4 and IPv6 entries for GitHub Pages;
  dropping IPv6 silently reduces availability for some networks.
