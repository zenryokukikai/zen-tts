// Model files are plain static files next to the page (see README: models/).
export function modelFetch(url: string): Promise<Response> {
  return fetch(url);
}
