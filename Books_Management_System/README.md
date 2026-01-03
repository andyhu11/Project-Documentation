# Books Management System

> A Microsoft Access (`.accdb`) based book management system for **stock-in, inventory, and sales statistics**.  
> Database file: `BooksManagementSystem.accdb`  
> Demo document: `BooksManagementSystem.pdf` (includes the full walkthrough and screenshots)

---

## 1. Project Overview

This project is built with **Microsoft Access** and consists of tables, queries, forms, and reports. It supports common workflows such as **stock-in (purchasing), inventory maintenance, sales registration, and sales analytics**.

Typical use cases:
- Small bookstores / classroom book corners for purchase & sales tracking
- Access forms/reports/query demonstration and coursework

---

## 2. Features

- **Book Management**
  - Add / edit / delete book records
  - Maintain book categories, authors, publishers, etc.
  - Inventory tracking / low-stock alerts (if implemented)

- **People & Basic Information Management (depends on your system design)**
  - Maintain master data such as: customers / employees / suppliers / system users, etc.

- **Stock-in (Purchase Registration)**
  - Record: purchase date, purchase quantity, purchase price (if applicable)
  - Inventory is automatically updated after stock-in (real-time inventory update logic)

- **Sales (Selling Registration)**
  - Record: sales date, quantity, unit price
  - Automatically deduct inventory after a sales order is saved (if implemented)
  - Supports printing invoices / receipts / sales lists (if implemented)

- **Query & Statistics**
  - Conditional search (keyword, category, date range, etc.)
  - Best-selling books ranking (e.g., “Most Popular Books” style query)
  - Sales amount / sales volume statistics by date range (e.g., yearly/monthly parameter queries)

- **Reports**
  - Print/export sales lists, inventory reports, and summary reports (if implemented)

---

## 3. Default Account

- Username: `admin`
- Password: `admin`

> For the login entry and UI steps, see the section “Information Login / Operation Selection” in `BooksManagementSystem.pdf`.

---

## 4. Environment Requirements

- **OS**: Windows 10/11 (recommended)
- **Software**:
  - Microsoft Access 2016/2019/2021 or Microsoft 365 Access  
  **or**
  - Microsoft Access Runtime (run-only, no design features)
- **File format**: `.accdb`

> For macOS: use a Windows VM or remote Windows environment.

---

## 5. Quick Start

1. Make sure the following files are in the **same folder**:
   - `BooksManagementSystem.accdb`
   - `BooksManagementSystem.pdf` (demo & operation guide)
2. Double-click to open: `BooksManagementSystem.accdb`
3. If you see a security warning:
   - Click **Enable Content**
   - Add the project folder to **Trusted Locations**
4. Log in with `admin/admin`, then select modules from the main menu (see the PDF for screenshots and step-by-step demo).

---

## 6. FAQ

### Q1: “Content disabled / Macros disabled” when opening
- Click **Enable Content** on the yellow security bar
- Or go to: `File -> Options -> Trust Center -> Trust Center Settings -> Trusted Locations` and add the project folder

### Q2: “Missing reference / VBA compile error”
- This may be caused by different Access/VBA library versions
- Open VBA editor: `Developer -> Visual Basic`
- Go to: `Tools -> References`, find any items marked **MISSING**, then uncheck or replace them with available references

### Q3: Can multiple people use it at the same time?
- A single `.accdb` file may have locking/conflict risks in multi-user scenarios. Recommended approach:
  - **Back-end**: tables only (shared drive/server)
  - **Front-end**: forms/queries/reports/code (one local copy per user)

---

## 7. Backup & Recovery

- **Backup**: simply copy `BooksManagementSystem.accdb`
- **Recommended**:
  - Back up before major changes (use dated filenames)
  - Run `Database Tools -> Compact & Repair Database` periodically

---

## 8. Project Structure

- `BooksManagementSystem.accdb` — main database file (tables/queries/forms/reports/macros/VBA)
- `BooksManagementSystem.pdf` — demo walkthrough & feature guide (kept in the same folder as the accdb)
